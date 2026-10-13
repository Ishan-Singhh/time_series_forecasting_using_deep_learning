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

Test-time analyses of a trained checkpoint (no training; writes preds/analysis/<name>.parquet):
    python -m bus_eta.models.stgnn --eval-only stgnn_mta --state-drop 0.5 --name stgnn_mta_drop50
    python -m bus_eta.models.stgnn --eval-only stgnn_mta --permute weather --name stgnn_mta_perm_weather
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
ATTR_COLS = ("seg_len_m", "n_signals", "lane_frac", "lane_active")  # nn_v2 token attributes


# ---------------------------------------------------------------- data
class Split:
    def __init__(self, name: str, drop_mta: bool, tok_names: list[str], drop_attrs: bool = False):
        z = np.load(NN / f"{name}.npz")
        self.bin = torch.from_numpy(z["bin"].astype(np.int64))
        self.route = torch.from_numpy(z["route"].astype(np.int64))
        self.dow = torch.from_numpy(z["dow"].astype(np.int64))
        self.issue_x = torch.from_numpy(z["issue_x"])
        tok_x = z["tok_x"].copy()
        if drop_mta:
            tok_x[..., [tok_names.index(c) for c in MTA_COLS]] = 0.0
        if drop_attrs:
            tok_x[..., [tok_names.index(c) for c in ATTR_COLS if c in tok_names]] = 0.0
        self.tok_x = torch.from_numpy(tok_x)
        self.tok_valid = torch.from_numpy(z["tok_valid"])
        self.tok_node = torch.from_numpy(z["tok_node"].astype(np.int64))
        # segment index for the *static* segment embedding; -1 = unknown. Differs from tok_node
        # (which also reads the live network state) only for held-out routes / spatial ablation.
        self.static_node = self.tok_node.clone()
        self.rt = torch.from_numpy(z["rt"])
        self.y = torch.from_numpy(z["y"])
        self.keys = (z["trip_id"], z["start_date"], z["k"])
        self._index_bins()

    FIELDS = ["bin", "route", "dow", "issue_x", "tok_x", "tok_valid", "tok_node", "static_node", "rt", "y"]

    def _index_bins(self) -> None:
        # issues are sorted by bin: contiguous [start, end) per bin
        b = self.bin.numpy()
        cut = np.flatnonzero(np.diff(b)) + 1
        self.bin_ranges = np.stack([np.concatenate([[0], cut]), np.concatenate([cut, [len(b)]])], 1)

    def drop_routes(self, routes: list[int]) -> None:
        """Remove every issue of the held-out routes (training / validation splits)."""
        keep = ~np.isin(self.route.numpy(), routes)
        k = torch.from_numpy(keep)
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f)[k])
        self.keys = tuple(a[keep] for a in self.keys)
        self._index_bins()

    def treat_as_unseen(self, routes: list[int], seen_nodes: np.ndarray) -> None:
        """Test issues of held-out routes: unknown route id, unknown static embedding for
        segments that never appeared in training. Live network state is still available."""
        held = torch.from_numpy(np.isin(self.route.numpy(), routes))
        self.route[held] = 0
        unseen = torch.from_numpy(~np.isin(self.static_node.numpy(), seen_nodes))
        self.static_node[held[:, None] & unseen] = -1

    def permute(self, field: str, cols, seed: int) -> None:
        """Shuffle a field (or some of its feature columns) across issues: permutation importance."""
        perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(self.bin)))
        x = getattr(self, field)
        if cols is None:
            setattr(self, field, x[perm])
        else:
            x[..., cols] = x[perm][..., cols]

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
        self.bin_map = None                                           # test-time permutation of bins
        self.train_drop = 0.0                                         # max share hidden while training

    def window(self, bins: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.bin_map is not None:
            bins = self.bin_map[bins]
        idx = (bins[:, None] + self.offsets).clamp(min=0)          # only completed bins < issue bin
        w = self.x[idx].float()                                      # (nb, W, N, 4)
        if training and self.train_drop > 0:
            # observation dropout: each bin in the batch hides a share p ~ U(0, train_drop) of its
            # (step, segment) observations, so the model also sees sparse network coverage
            p = torch.rand(w.shape[0], 1, 1, device=w.device) * self.train_drop
            w = w * (torch.rand(w.shape[:3], device=w.device) >= p)[..., None]
        return w

    def drop_observations(self, p: float, seed: int) -> None:
        """Hide a random share p of (bin, segment) observations, as if fewer buses reported."""
        g = torch.Generator(device=self.x.device).manual_seed(seed)
        for s in range(0, self.x.shape[0], 4096):                    # chunks keep the mask small
            chunk = self.x[s:s + 4096]
            hide = torch.rand(chunk.shape[:2], generator=g, device=self.x.device) < p
            chunk[hide] = 0.0                                         # values and observed-mask -> unobserved

    def permute_bins(self, test_bins: np.ndarray, seed: int) -> None:
        """Feed each test bin the network state of another (random) test bin."""
        u = np.unique(test_bins)
        perm = np.random.default_rng(seed).permutation(u)
        m = torch.arange(self.x.shape[0], device=self.x.device)
        m[torch.from_numpy(u).to(m.device)] = torch.from_numpy(perm).to(m.device)
        self.bin_map = m


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
        if adaptive is not None:
            out = out + self.proj[-1](torch.einsum("nm,bmd->bnd", adaptive, h))
        return self.norm(h + self.drop(F.gelu(out)))


class GraphEncoder(nn.Module):
    def __init__(self, n_nodes, d, n_layers, dropout, supports, node_feats=None):
        """node_feats (N, F) given = inductive variant: segments are described by their
        physical attributes instead of learned per-segment vectors, and there is no learned
        adaptive adjacency, so the encoder transfers to segments never seen in training."""
        super().__init__()
        self.supports = supports
        self.inp = nn.Linear(4, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.inductive = node_feats is not None
        if self.inductive:
            self.register_buffer("node_feats", node_feats)
            self.feat_proj = nn.Linear(node_feats.shape[1], d)
        else:
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
        if self.inductive:
            h = hn[-1].view(nb, n, -1) + self.feat_proj(self.node_feats)
            adaptive = None
        else:
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


def load_node_feats(device, which="all") -> torch.Tensor:
    """Standardised static segment attributes (segment_attrs.py) for the inductive encoder.
    which: all | len (segment length only) | none (a constant: anonymous segments), the
    last two being controls for how much the signal and bus-lane attributes contribute."""
    a = pl.read_parquet(GRAPH / "node_attrs.parquet").sort("node")
    x = a.select("seg_len_m", "n_signals", "signals_per_km", "lane_ded_frac", "lane_shared_frac").to_numpy()
    x = np.column_stack([np.log1p(x[:, 0]), x[:, 1:]])
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    if which == "len":
        x = x[:, :1]
    elif which == "none":
        x = np.zeros((len(x), 1))
    return torch.tensor(x, dtype=torch.float32, device=device)


def make_encoder(kind, n_nodes, d, dropout, supports, node_feats=None):
    if kind == "dcrnn":
        return DCRNNEncoder(n_nodes, d, 2, dropout, supports)
    if kind == "gwnet":
        return GWNetEncoder(n_nodes, d, dropout, supports)
    return GraphEncoder(n_nodes, d, 2, dropout, supports, node_feats)


class STETA(nn.Module):
    def __init__(self, n_nodes, n_routes, f_issue, f_tok, supports, use_graph, d=128, d_g=64, dropout=0.1,
                 encoder="ours", node_feats=None, quantiles=None):
        super().__init__()
        self.n_nodes, self.use_graph = n_nodes, use_graph
        self.quantiles = list(quantiles) if quantiles else None
        self.n_out = len(quantiles) if quantiles else 1
        self.issue = nn.Sequential(nn.Linear(f_issue, d), nn.GELU(), nn.Linear(d, d))
        self.route = nn.Embedding(n_routes + 1, d)
        self.dow = nn.Embedding(7, d)
        self.tok = nn.Linear(f_tok, d)
        self.hpos = nn.Embedding(H, d)
        self.node_static = nn.Embedding(n_nodes + 1, d)               # last row = unknown segment
        if use_graph:
            self.graph = make_encoder(encoder, n_nodes, d_g, dropout, supports, node_feats)
            self.gproj = nn.Linear(d_g, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 3)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.n_out))

    def forward(self, b, state: NetworkState | None):
        node = torch.where(b["tok_node"] >= 0, b["tok_node"], torch.full_like(b["tok_node"], self.n_nodes))
        static = torch.where(b["static_node"] >= 0, b["static_node"], torch.full_like(b["static_node"], self.n_nodes))
        t = self.tok(b["tok_x"]) + self.hpos.weight[None] + self.node_static(static)
        if self.use_graph:
            ubins, slot = torch.unique(b["bin"], return_inverse=True)
            g = self.graph(state.window(ubins, self.training)).to(t.dtype)            # (nb, N, d_g)
            flat_idx = slot[:, None] * self.n_nodes + node.clamp(max=self.n_nodes - 1)
            g_tok = g.reshape(-1, g.shape[-1])[flat_idx] * (b["tok_node"] >= 0)[..., None]
            t = t + self.gproj(g_tok)
        q = (self.issue(b["issue_x"]) + self.route(b["route"]) + self.dow(b["dow"]))[:, None]
        valid = torch.cat([torch.ones_like(b["tok_valid"][:, :1]), b["tok_valid"]], 1)
        out = self.enc(torch.cat([q, t], 1), src_key_padding_mask=~valid)
        o = self.head(out[:, 1:]) * RESIDUAL_SCALE                    # (B, H, n_out)
        return o.squeeze(-1) if self.n_out == 1 else o


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
    return {n: getattr(split, n)[i].to(device, non_blocking=True) for n in Split.FIELDS}


def median(pred, quantiles):
    """Point prediction: the median column of a quantile head, or the single output."""
    return pred if pred.dim() == 2 else pred[..., quantiles.index(0.5)]


def masked_l1(pred, b):
    y = b["y"]
    m = torch.isfinite(y) & b["tok_valid"]
    return (torch.abs(b["rt"] + pred - torch.nan_to_num(y)) * m).sum() / m.sum().clamp(min=1), m.sum()


def masked_pinball(pred, b, quantiles):
    """Mean pinball loss summed over quantiles (= L1 / 2 per quantile at q = 0.5)."""
    y = b["y"]
    m = (torch.isfinite(y) & b["tok_valid"])[..., None]
    err = torch.nan_to_num(y)[..., None] - (b["rt"][..., None] + pred)
    q = torch.tensor(quantiles, device=pred.device)
    loss = torch.maximum(q * err, (q - 1) * err) * m
    return loss.sum() / m.sum().clamp(min=1), m.sum()


@torch.no_grad()
def evaluate_split(model, split, state, device, batch_size):
    model.eval()
    tot, n = 0.0, 0
    for idx in split.batches(batch_size, False, np.random.default_rng(0)):
        b = to_device(split, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(b, state).float()
        loss, cnt = masked_l1(median(pred, model.quantiles), b)
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
            out = model(b, state).float()
        point = (b["rt"] + median(out, model.quantiles)).cpu().numpy()
        valid = b["tok_valid"].cpu().numpy()
        r, h = np.nonzero(valid)
        cols = {"trip_id": split.keys[0][idx][r], "start_date": split.keys[1][idx][r],
                "k": split.keys[2][idx][r], "h": (h + 1).astype(np.int8), "pred": point[r, h]}
        if model.quantiles:                                           # q10, q50, q90, ... columns
            qs = (b["rt"][..., None] + out).cpu().numpy()
            cols |= {f"q{round(100 * q)}": qs[r, h, i] for i, q in enumerate(model.quantiles)}
        parts.append(pl.DataFrame(cols))
    return pl.concat(parts)


ISSUE_GROUPS = {
    "bus_state": ["delay_dep", "delay_arr_k1", "delay_arr_k2", "delay_arr_k3", "pace_ratio_3", "headway_s",
                  "issue_lag", "trip_progress", "k", "n_stops"],
    "time": ["tod_sin", "tod_cos"],
    "calendar_weather": ["is_holiday", "n_major_events", "temp_c", "precip_mm", "snow_cm", "wind_kmh"],
    "school": ["is_school_day"],                                     # nn_v2 only
}
TOK_GROUPS = {"schedule": ["dist", "run_sched", "sched"], "rt": ["rt", "rt_fresh"], "mta": ["mta", "has_mta"],
              "segment_attrs": list(ATTR_COLS)}                      # nn_v2 only
PERMUTE_GROUPS = list(ISSUE_GROUPS) + list(TOK_GROUPS) + ["route", "dow", "graph"]


def eval_only(args) -> None:
    """Score a trained checkpoint on test after a perturbation (missing state or a permuted input)."""
    src = json.loads((MODELS / f"{args.eval_only}.json").read_text())["args"]
    global NN
    NN = PROCESSED / src.get("data", "nn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((NN / "stats.json").read_text())
    issue_names, tok_names = meta["issue_num"], meta["tok_num"]
    test = Split("test", src["no_mta"], tok_names, src.get("no_attrs", False))
    state = None if src["no_graph"] else NetworkState(device)
    supports = [] if src["no_graph"] else load_supports(device)
    n_nodes = pl.read_parquet(GRAPH / "nodes.parquet").height
    model = STETA(n_nodes, len(meta["route_map"]), test.issue_x.shape[1], test.tok_x.shape[2], supports,
                  use_graph=not src["no_graph"], encoder=src.get("encoder", "ours"),
                  node_feats=load_node_feats(device, src.get("inductive_feats", "all")) if src.get("inductive") else None,
                  quantiles=src.get("quantiles") or None).to(device)
    model.load_state_dict(torch.load(MODELS / f"{args.eval_only}.pt", map_location=device))
    if src.get("holdout_routes"):
        cfg = json.loads((MODELS / f"{args.eval_only}.json").read_text())
        test.treat_as_unseen([meta["route_map"][r] for r in src["holdout_routes"]], np.array(cfg["seen_nodes"]))
    if args.state_drop > 0:
        assert state is not None, "--state-drop needs a graph model"
        state.drop_observations(args.state_drop, args.seed)
    if args.permute:
        g = args.permute
        if g in ISSUE_GROUPS:
            test.permute("issue_x", [issue_names.index(c) for c in ISSUE_GROUPS[g] if c in issue_names], args.seed)
        elif g in TOK_GROUPS:
            test.permute("tok_x", [tok_names.index(c) for c in TOK_GROUPS[g] if c in tok_names], args.seed)
        elif g in ("route", "dow"):
            test.permute(g, None, args.seed)
        else:
            assert state is not None, "--permute graph needs a graph model"
            state.permute_bins(test.bin.numpy(), args.seed)
    bs = args.batch_size if args.batch_size > 0 else auto_batch_size(device)
    mae = evaluate_split(model, test, state, device, 4 * bs)
    print(f"{args.name}: test MAE {mae:.2f}s (from {args.eval_only}, state_drop={args.state_drop}, "
          f"permute={args.permute})", flush=True)
    out = PREDS / "analysis"              # kept out of the main table; evaluate with --models analysis/<name>
    out.mkdir(parents=True, exist_ok=True)
    predict(model, test, state, device, 4 * bs).write_parquet(out / f"{args.name}.parquet")


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
    p.add_argument("--data", default="nn", help="packed data folder under data/processed (nn, nn_v2)")
    p.add_argument("--no-attrs", action="store_true", help="zero the nn_v2 segment-attribute token columns")
    p.add_argument("--inductive", action="store_true",
                   help="graph encoder from segment attributes, no per-segment vectors / adaptive adjacency")
    p.add_argument("--inductive-feats", choices=["all", "len", "none"], default="all",
                   help="attributes the inductive encoder sees (len/none are controls)")
    p.add_argument("--holdout-routes", nargs="*", default=[], help="routes removed from train/val (cold start)")
    p.add_argument("--ablate-p", type=float, default=0.0, help="spatial input ablation probability (BusTr)")
    p.add_argument("--quantiles", type=float, nargs="*", default=[],
                   help="quantile head trained with pinball loss (must include 0.5, the point ETA)")
    p.add_argument("--train-state-drop", type=float, default=0.0,
                   help="observation dropout while training: hide up to this share of network state")
    p.add_argument("--eval-only", metavar="MODEL", help="score this trained checkpoint on test instead of training")
    p.add_argument("--state-drop", type=float, default=0.0, help="eval-only: share of network observations hidden")
    p.add_argument("--permute", choices=PERMUTE_GROUPS, help="eval-only: input group shuffled across issues")
    args = p.parse_args()
    assert not args.quantiles or 0.5 in args.quantiles, "--quantiles must include 0.5"
    args.quantiles = sorted(args.quantiles)
    if args.eval_only:
        torch.set_num_threads(n_cpus())
        return eval_only(args)
    global NN
    NN = PROCESSED / args.data

    torch.manual_seed(args.seed)
    torch.set_num_threads(n_cpus())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_size <= 0:
        args.batch_size = auto_batch_size(device)
    meta = json.loads((NN / "stats.json").read_text())
    tok_names = meta["tok_num"]
    train, val, test = (Split(s, args.no_mta, tok_names, args.no_attrs) for s in ("train", "val", "test"))
    if args.holdout_routes:
        held = [meta["route_map"][r] for r in args.holdout_routes]
        train.drop_routes(held)
        val.drop_routes(held)
        assert not np.isin(train.route.numpy(), held).any()
        seen = np.unique(train.tok_node.numpy()[train.tok_valid.numpy()])
        n_held = int(np.isin(test.route.numpy(), held).sum())
        test.treat_as_unseen(held, seen)
        print(f"held out {args.holdout_routes}: {n_held:,} test issues on unseen routes", flush=True)
    else:
        seen = None
    state = None if args.no_graph else NetworkState(device)
    if state is not None:
        state.train_drop = args.train_state_drop
    supports = [] if args.no_graph else load_supports(device)
    n_nodes = pl.read_parquet(GRAPH / "nodes.parquet").height
    model = STETA(n_nodes, len(meta["route_map"]), train.issue_x.shape[1], train.tok_x.shape[2],
                  supports, use_graph=not args.no_graph, encoder=args.encoder,
                  node_feats=load_node_feats(device, args.inductive_feats) if args.inductive else None,
                  quantiles=args.quantiles or None).to(device)
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
            if args.ablate_p > 0:  # BusTr-style spatial input ablation: whole issue loses route + static ids
                drop = torch.rand(len(idx), device=device) < args.ablate_p
                b["route"] = torch.where(drop, torch.zeros_like(b["route"]), b["route"])
                b["static_node"] = torch.where(drop[:, None], torch.full_like(b["static_node"], -1), b["static_node"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(b, state).float()
            loss, n = masked_pinball(pred, b, args.quantiles) if args.quantiles else masked_l1(pred, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * n.item()
            cnt += n.item()
        val_mae = evaluate_split(model, val, state, device, 4 * args.batch_size)
        print(f"epoch {epoch:2d} train {'pinball' if args.quantiles else 'MAE'} {run / cnt:6.2f}s  val MAE {val_mae:6.2f}s  ({time.time() - t0:.0f}s)",
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
    out = PREDS / "analysis" if args.holdout_routes else PREDS   # cold-start runs stay out of the main table
    out.mkdir(parents=True, exist_ok=True)
    predict(model, test, state, device, 4 * args.batch_size).write_parquet(out / f"{args.name}.parquet")
    (MODELS / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "best_val_mae": best,
                                                          "seen_nodes": None if seen is None else seen.tolist()}))
    print("wrote test predictions", flush=True)


if __name__ == "__main__":
    main()
