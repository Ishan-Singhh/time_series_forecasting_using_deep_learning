"""Spatio-temporal graph ETA model (and its no-graph ablation).

Graph encoder (once per 5-min bin, shared by all buses issued in that bin):
  last WINDOW completed bins of network state (speed, #obs, dwell, observed-mask) per
  segment node -> per-node GRU over time -> graph layers mixing flow-forward,
  flow-backward, corridor and a learned adaptive adjacency (Graph WaveNet-style).

Trip decoder (per issue):
  [issue token] + H downstream tokens. Each token = per-horizon features + horizon
  embedding + static segment embedding + the live graph embedding of the segment it
  crosses. A Transformer encoder mixes them; the head outputs a correction to the
  real-time profile ETA for every horizon. Loss: masked L1 against observed arrivals.

Batches are whole time bins, so the graph is encoded once per bin. Early stopping uses
the validation month only.

    python -m bus_eta.models.stgnn --name stgnn_mta
    python -m bus_eta.models.stgnn --name stgnn_no_mta --no-mta
    python -m bus_eta.models.stgnn --name seq_mta --no-graph
    python -m bus_eta.models.stgnn --name dcrnn_mta --encoder dcrnn     # or gwnet
"""
import argparse
import json
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from bus_eta.config import H, INTERIM, PROCESSED, n_cpus
from bus_eta.evaluate import PREDS

NN = PROCESSED / "nn"
STATE = PROCESSED / "network_state"
GRAPH = INTERIM / "graph"
MODELS = PROCESSED / "models"
WINDOW = 12          # 1 hour of 5-min bins
RESIDUAL_SCALE = 60.0
MAX_BINS_PER_BATCH = 16  # graph is encoded once per bin; night bins are sparse, so cap bins too
MTA_COLS = ("mta", "has_mta")


# ---------------------------------------------------------------- data
class Split:
    def __init__(self, name: str, drop_mta: bool, tok_names: list[str]):
        z = np.load(NN / f"{name}.npz")
        self.bin = torch.from_numpy(z["bin"].astype(np.int64))
        self.route = torch.from_numpy(z["route"].astype(np.int64))
        self.dow = torch.from_numpy(z["dow"].astype(np.int64))
        self.issue_x = torch.from_numpy(z["issue_x"])
        tok_x = z["tok_x"].copy()
        if drop_mta:
            tok_x[..., [tok_names.index(c) for c in MTA_COLS]] = 0.0
        self.tok_x = torch.from_numpy(tok_x)
        self.tok_valid = torch.from_numpy(z["tok_valid"])
        self.tok_node = torch.from_numpy(z["tok_node"].astype(np.int64))
        self.rt = torch.from_numpy(z["rt"])
        self.y = torch.from_numpy(z["y"])
        self.keys = (z["trip_id"], z["start_date"], z["k"])
        # issues are sorted by bin: contiguous [start, end) per bin
        b = z["bin"]
        cut = np.flatnonzero(np.diff(b)) + 1
        self.bin_ranges = np.stack([np.concatenate([[0], cut]), np.concatenate([cut, [len(b)]])], 1)

    def batches(self, target_size: int, shuffle: bool, rng: np.random.Generator, max_bins: int = MAX_BINS_PER_BATCH):
        order = rng.permutation(len(self.bin_ranges)) if shuffle else np.arange(len(self.bin_ranges))
        cur, n = [], 0
        for i in order:
            s, e = self.bin_ranges[i]
            cur.append(np.arange(s, e))
            n += e - s
            if n >= target_size or len(cur) >= max_bins:
                yield np.concatenate(cur)
                cur, n = [], 0
        if cur:
            yield np.concatenate(cur)


class NetworkState:
    """State tensor on the GPU, normalised, with an observed-mask channel."""

    def __init__(self, device):
        meta = json.loads((STATE / "meta.json").read_text())
        raw = np.load(STATE / "state.npy").astype(np.float32)
        obs = ~np.isnan(raw[..., 0:1])
        x = np.concatenate([np.nan_to_num(raw[..., 0:1] / 10.0), np.log1p(np.nan_to_num(raw[..., 1:2])),
                            np.nan_to_num(raw[..., 2:3] / 30.0), obs.astype(np.float32)], -1)
        self.x = torch.from_numpy(x).to(device=device, dtype=torch.float16)
        self.n_nodes = meta["n_nodes"]
        self.offsets = torch.arange(-WINDOW, 0, device=device)

    def window(self, bins: torch.Tensor) -> torch.Tensor:
        idx = (bins[:, None] + self.offsets).clamp(min=0)          # only completed bins < issue bin
        return self.x[idx].float()                                   # (nb, W, N, 4)


def sparse_support(src, dst, n, device):
    """Row-normalised sparse adjacency with self loops: row i averages over i's neighbours."""
    src = np.concatenate([src, np.arange(n)])
    dst = np.concatenate([dst, np.arange(n)])
    deg = np.bincount(dst, minlength=n).astype(np.float32)
    vals = 1.0 / deg[dst]
    return torch.sparse_coo_tensor(np.stack([dst, src]), vals, (n, n)).coalesce().to(device)


def load_supports(device):
    e = pl.read_parquet(GRAPH / "edges.parquet")
    n = pl.read_parquet(GRAPH / "nodes.parquet").height
    flow = e.filter(pl.col("edge_type") == "flow")
    cor = e.filter(pl.col("edge_type") == "corridor")
    fs, fd = flow["src"].to_numpy().astype(np.int64), flow["dst"].to_numpy().astype(np.int64)
    return [sparse_support(fs, fd, n, device),                       # upstream -> downstream
            sparse_support(fd, fs, n, device),                       # downstream -> upstream (spillback)
            sparse_support(cor["src"].to_numpy().astype(np.int64), cor["dst"].to_numpy().astype(np.int64), n, device)]


# ---------------------------------------------------------------- model
class GraphLayer(nn.Module):
    def __init__(self, d, n_supports, dropout):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, d) for _ in range(n_supports + 1)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, sparse_supports, adaptive):
        nb, n, d = h.shape
        flat = h.transpose(0, 1).reshape(n, nb * d)
        out = self.proj[0](h)
        for proj, s in zip(self.proj[1:], sparse_supports):
            out = out + proj(torch.sparse.mm(s, flat).reshape(n, nb, d).transpose(0, 1))
        out = out + self.proj[-1](torch.einsum("nm,bmd->bnd", adaptive, h))
        return self.norm(h + self.drop(F.gelu(out)))


class GraphEncoder(nn.Module):
    def __init__(self, n_nodes, d, n_layers, dropout, supports):
        super().__init__()
        self.supports = supports
        self.inp = nn.Linear(4, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, d) * 0.02)
        self.adp_src = nn.Parameter(torch.randn(n_nodes, 16) * 0.1)
        self.adp_dst = nn.Parameter(torch.randn(n_nodes, 16) * 0.1)
        self.layers = nn.ModuleList([GraphLayer(d, len(supports) + 1, dropout) for _ in range(n_layers)])

    @torch.autocast("cuda", enabled=False)  # sparse mm / GRU in fp32; this block is small
    def forward(self, x):                                             # (nb, W, N, 4)
        x = x.float()
        nb, w, n, _ = x.shape
        h = self.inp(x).permute(0, 2, 1, 3).reshape(nb * n, w, -1)
        _, hn = self.gru(h)
        h = hn[-1].view(nb, n, -1) + self.node_emb
        adaptive = torch.softmax(F.relu(self.adp_src @ self.adp_dst.T), dim=1)
        for layer in self.layers:
            h = layer(h, self.supports, adaptive)
        return h                                                      # (nb, N, d)


def diffuse(x, supports, k, dim):
    """[x, S1 x, ..., S1^k x, S2 x, ...]: k-step diffusion of x along node axis `dim` per support."""
    xt = x.movedim(dim, 0)
    shape = xt.shape
    flat = xt.reshape(shape[0], -1)
    out = [x]
    for s in supports:
        h = flat
        for _ in range(k):
            h = torch.sparse.mm(s, h) if s.is_sparse else s @ h
            out.append(h.reshape(shape).movedim(0, dim))
    return out


class DiffusionConv(nn.Module):
    """DCRNN graph convolution: linear map of the concatenated diffusion terms."""

    def __init__(self, c_in, c_out, n_supports, k):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(c_in * (1 + n_supports * k), c_out)

    def forward(self, x, supports):                                  # (nb, N, c_in)
        return self.lin(torch.cat(diffuse(x, supports, self.k, 1), -1))


class DCGRUCell(nn.Module):
    def __init__(self, c_in, d, n_supports, k):
        super().__init__()
        self.gates = DiffusionConv(c_in + d, 2 * d, n_supports, k)
        self.cand = DiffusionConv(c_in + d, d, n_supports, k)

    def forward(self, x, h, supports):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1), supports)).chunk(2, -1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), supports))
        return u * h + (1 - u) * c


class DCRNNEncoder(nn.Module):
    """DCRNN encoder (Li et al., ICLR 2018): stacked DCGRU cells over the WINDOW bins with
    K-step diffusion along flow-forward / flow-backward segments (plus corridor edges).
    Our trip decoder replaces DCRNN's seq2seq decoder."""

    def __init__(self, n_nodes, d, n_layers, dropout, supports, k=2):
        super().__init__()
        self.supports, self.d = supports, d
        self.cells = nn.ModuleList([DCGRUCell(4 if i == 0 else d, d, len(supports), k) for i in range(n_layers)])
        self.node_emb = nn.Parameter(torch.randn(n_nodes, d) * 0.02)
        self.drop = nn.Dropout(dropout)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):                                             # (nb, W, N, 4)
        x = x.float()
        nb, w, n, _ = x.shape
        hs = [x.new_zeros(nb, n, self.d) for _ in self.cells]
        for t in range(w):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                hs[i] = cell(inp, hs[i], self.supports)
                inp = self.drop(hs[i])
        return hs[-1] + self.node_emb                                 # (nb, N, d)


class GWNetEncoder(nn.Module):
    """Graph WaveNet encoder (Wu et al., IJCAI 2019): gated dilated causal temporal convs
    (dilations 1,2 x4, receptive field 13) interleaved with diffusion GCNs over the fixed
    supports and a self-adaptive adjacency; skip connections summed, last step returned."""

    def __init__(self, n_nodes, d, dropout, supports, c_res=32, c_skip=128, k=2, dilations=(1, 2) * 4):
        super().__init__()
        self.supports, self.k, self.drop = supports, k, dropout
        self.rf = 1 + sum(dilations)
        self.nodevec1 = nn.Parameter(torch.randn(n_nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, n_nodes))
        self.start = nn.Conv2d(4, c_res, 1)
        self.filt = nn.ModuleList([nn.Conv2d(c_res, c_res, (1, 2), dilation=(1, dl)) for dl in dilations])
        self.gate = nn.ModuleList([nn.Conv2d(c_res, c_res, (1, 2), dilation=(1, dl)) for dl in dilations])
        self.skip = nn.ModuleList([nn.Conv2d(c_res, c_skip, 1) for _ in dilations])
        n_sup = len(supports) + 1
        self.gcn = nn.ModuleList([nn.Conv2d(c_res * (1 + n_sup * k), c_res, 1) for _ in dilations])
        self.bn = nn.ModuleList([nn.BatchNorm2d(c_res) for _ in dilations])
        self.end = nn.Sequential(nn.ReLU(), nn.Conv2d(c_skip, c_skip, 1), nn.ReLU(), nn.Conv2d(c_skip, d, 1))
        self.node_emb = nn.Parameter(torch.randn(n_nodes, d) * 0.02)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):                                             # (nb, W, N, 4)
        x = x.float().permute(0, 3, 2, 1)                             # (nb, 4, N, W)
        x = F.pad(x, (self.rf - x.shape[-1], 0)) if x.shape[-1] < self.rf else x
        adaptive = torch.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        supports = list(self.supports) + [adaptive]
        h = self.start(x)
        skip = 0
        for filt, gate, sk, gcn, bn in zip(self.filt, self.gate, self.skip, self.gcn, self.bn):
            res = h
            h = torch.tanh(filt(h)) * torch.sigmoid(gate(h))
            s = sk(h[..., -1:])
            skip = s + (skip if isinstance(skip, int) else skip[..., -1:])
            h = gcn(torch.cat(diffuse(h, supports, self.k, 2), 1))
            h = F.dropout(h, self.drop, self.training)
            h = bn(h + res[..., -h.shape[-1]:])
        out = self.end(skip)[..., -1]                                 # (nb, d, N)
        return out.transpose(1, 2) + self.node_emb                    # (nb, N, d)


ENCODERS = ("ours", "dcrnn", "gwnet")


def make_encoder(kind, n_nodes, d, dropout, supports):
    if kind == "dcrnn":
        return DCRNNEncoder(n_nodes, d, 2, dropout, supports)
    if kind == "gwnet":
        return GWNetEncoder(n_nodes, d, dropout, supports)
    return GraphEncoder(n_nodes, d, 2, dropout, supports)


class STETA(nn.Module):
    def __init__(self, n_nodes, n_routes, f_issue, f_tok, supports, use_graph, d=128, d_g=64, dropout=0.1,
                 encoder="ours"):
        super().__init__()
        self.n_nodes, self.use_graph = n_nodes, use_graph
        self.issue = nn.Sequential(nn.Linear(f_issue, d), nn.GELU(), nn.Linear(d, d))
        self.route = nn.Embedding(n_routes + 1, d)
        self.dow = nn.Embedding(7, d)
        self.tok = nn.Linear(f_tok, d)
        self.hpos = nn.Embedding(H, d)
        self.node_static = nn.Embedding(n_nodes + 1, d)               # last row = unknown segment
        if use_graph:
            self.graph = make_encoder(encoder, n_nodes, d_g, dropout, supports)
            self.gproj = nn.Linear(d_g, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 3)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b, state: NetworkState | None):
        node = torch.where(b["tok_node"] >= 0, b["tok_node"], torch.full_like(b["tok_node"], self.n_nodes))
        t = self.tok(b["tok_x"]) + self.hpos.weight[None] + self.node_static(node)
        if self.use_graph:
            ubins, slot = torch.unique(b["bin"], return_inverse=True)
            g = self.graph(state.window(ubins)).to(t.dtype)            # (nb, N, d_g)
            flat_idx = slot[:, None] * self.n_nodes + node.clamp(max=self.n_nodes - 1)
            g_tok = g.reshape(-1, g.shape[-1])[flat_idx] * (b["tok_node"] >= 0)[..., None]
            t = t + self.gproj(g_tok)
        q = (self.issue(b["issue_x"]) + self.route(b["route"]) + self.dow(b["dow"]))[:, None]
        valid = torch.cat([torch.ones_like(b["tok_valid"][:, :1]), b["tok_valid"]], 1)
        out = self.enc(torch.cat([q, t], 1), src_key_padding_mask=~valid)
        return self.head(out[:, 1:]).squeeze(-1) * RESIDUAL_SCALE


# ---------------------------------------------------------------- training
def auto_batch_size(device) -> int:
    """Size batches to the GPU memory free right now (the GPUs are shared with other users)."""
    if device.type != "cuda":
        return 512
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    size = 512 if free_gb < 6 else 1024 if free_gb < 12 else 2048 if free_gb < 24 else 4096
    print(f"GPU free memory {free_gb:.1f} GiB -> batch size {size}", flush=True)
    return size


def to_device(split: Split, idx: np.ndarray, device) -> dict:
    i = torch.from_numpy(idx)
    names = ["bin", "route", "dow", "issue_x", "tok_x", "tok_valid", "tok_node", "rt", "y"]
    return {n: getattr(split, n)[i].to(device, non_blocking=True) for n in names}


def masked_l1(pred, b):
    y = b["y"]
    m = torch.isfinite(y) & b["tok_valid"]
    return (torch.abs(b["rt"] + pred - torch.nan_to_num(y)) * m).sum() / m.sum().clamp(min=1), m.sum()


@torch.no_grad()
def evaluate_split(model, split, state, device, batch_size):
    model.eval()
    tot, n = 0.0, 0
    for idx in split.batches(batch_size, False, np.random.default_rng(0)):
        b = to_device(split, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(b, state).float()
        loss, cnt = masked_l1(pred, b)
        tot += loss.item() * cnt.item()
        n += cnt.item()
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, split, state, device, batch_size) -> pl.DataFrame:
    model.eval()
    parts = []
    for idx in split.batches(batch_size, False, np.random.default_rng(0)):
        b = to_device(split, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = (b["rt"] + model(b, state).float()).cpu().numpy()
        valid = b["tok_valid"].cpu().numpy()
        r, h = np.nonzero(valid)
        parts.append(pl.DataFrame({
            "trip_id": split.keys[0][idx][r], "start_date": split.keys[1][idx][r],
            "k": split.keys[2][idx][r], "h": (h + 1).astype(np.int8), "pred": pred[r, h]}))
    return pl.concat(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--no-mta", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=0, help="0 = choose from free GPU memory")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", choices=ENCODERS, default="ours", help="graph encoder (ignored with --no-graph)")
    p.add_argument("--max-bins", type=int, default=MAX_BINS_PER_BATCH, help="time bins per batch (graph memory)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(n_cpus())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_size <= 0:
        args.batch_size = auto_batch_size(device)
    meta = json.loads((NN / "stats.json").read_text())
    tok_names = meta["tok_num"]
    train, val, test = (Split(s, args.no_mta, tok_names) for s in ("train", "val", "test"))
    state = None if args.no_graph else NetworkState(device)
    supports = [] if args.no_graph else load_supports(device)
    n_nodes = pl.read_parquet(GRAPH / "nodes.parquet").height
    model = STETA(n_nodes, len(meta["route_map"]), train.issue_x.shape[1], train.tok_x.shape[2],
                  supports, use_graph=not args.no_graph, encoder=args.encoder).to(device)
    print(f"{args.name}: {sum(p.numel() for p in model.parameters()):,} params on {device}; "
          f"train {len(train.bin):,} issues, val {len(val.bin):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = sum(1 for _ in train.batches(args.batch_size, False, np.random.default_rng(0), args.max_bins))
    # shuffled bin grouping can yield slightly more batches than the unshuffled count
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, pct_start=0.05,
                                                total_steps=args.epochs * int(steps_per_epoch * 1.05 + 10))
    rng = np.random.default_rng(args.seed)
    best, bad, ckpt = math.inf, 0, MODELS / f"{args.name}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, cnt = time.time(), 0.0, 0
        for idx in train.batches(args.batch_size, True, rng, args.max_bins):
            b = to_device(train, idx, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(b, state).float()
            loss, n = masked_l1(pred, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * n.item()
            cnt += n.item()
        val_mae = evaluate_split(model, val, state, device, 4 * args.batch_size)
        print(f"epoch {epoch:2d} train MAE {run / cnt:6.2f}s  val MAE {val_mae:6.2f}s  ({time.time() - t0:.0f}s)",
              flush=True)
        if val_mae < best - 0.01:
            best, bad = val_mae, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop; best val MAE {best:.2f}s", flush=True)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    PREDS.mkdir(parents=True, exist_ok=True)
    predict(model, test, state, device, 4 * args.batch_size).write_parquet(PREDS / f"{args.name}.parquet")
    (MODELS / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "best_val_mae": best}))
    print("wrote test predictions", flush=True)


if __name__ == "__main__":
    main()
