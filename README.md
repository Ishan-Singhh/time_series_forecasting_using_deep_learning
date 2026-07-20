



COMPLETE WORKFLOW




                    RAW DATA
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    GTFS Static    GTFS-RT      External Data
                                   │
                           Weather
                           Traffic
                           Calendar
                           Ridership
                       Service Alerts
                       Road Network
                       │
                       ▼
             Data Cleaning & Validation
                       │
        Remove duplicates
        Missing values
        Timestamp synchronization
        Map matching
        Remove bad trips
                       │
                       ▼
              Feature Engineering
                       │
       Temporal Features
       Spatial Features
       Operational Features
       Context Features
       Traffic Features
                       │
                       ▼
             Graph Construction
                       │
      Nodes = Bus Stops
      Edges = Road/Route Connections
      Node Features
      Edge Features
                       │
                       ▼
             Train / Val / Test Split
                       │
       Baseline Models
       ├── Historical Average
       ├── Linear Regression
       ├── Random Forest
       ├── XGBoost
                       │
              Compare Results
                       │
                       ▼
           Deep Learning Models
       ├── LSTM
       ├── GCN
       └── GAT (Final Model)
                       │
                       ▼
              Hyperparameter Tuning
                       │
                       ▼
                Model Evaluation
                       │
         MAE
         RMSE
         MAPE
         R²
                       │
                       ▼
             Ablation Experiments
                       │
       Remove Weather
       Remove Traffic
       Remove Graph
       Remove Attention
                       │
                       ▼
               Error Analysis
                       │
            Peak vs Off-Peak
            Rain vs Clear
           Weekday vs Weekend
            Different Routes
                       │
                       ▼
             Figures & Tables
                       │
                       ▼
                Paper Writing



PROJECT STRUCTURE


     Bus-Travel-Time-Prediction/

    │
    ├── data/
    │   ├── raw/
    │   │     gtfs_static/
    │   │     gtfs_rt/
    │   │     weather/
    │   │     traffic/
    │   │     calendar/
    │   │
    │   ├── interim/
    │   │
    │   ├── processed/
    │   │
    │   └── graph/
    │
    ├── notebooks/
    │   │
    │   ├── 01_exploration.ipynb
    │   ├── 02_feature_analysis.ipynb
    │   └── 03_visualization.ipynb
    │
    ├── src/
    │
    │   ├── preprocessing/
    │   │     load_data.py
    │   │     clean_data.py
    │   │     synchronize.py
    │   │     map_matching.py
    │   │
    │   ├── features/
    │   │     temporal.py
    │   │     spatial.py
    │   │     operational.py
    │   │     weather.py
    │   │     traffic.py
    │   │     calendar.py
    │   │     ridership.py
    │   │
    │   ├── graph/
    │   │     graph_builder.py
    │   │     node_features.py
    │   │     edge_features.py
    │   │
    │   ├── models/
    │   │     baseline.py
    │   │     random_forest.py
    │   │     xgboost.py
    │   │     lstm.py
    │   │     gat.py
    │   │
    │   ├── training/
    │   │     train.py
    │   │     validate.py
    │   │     test.py
    │   │
    │   ├── evaluation/
    │   │     metrics.py
    │   │     plots.py
    │   │     ablation.py
    │   │     error_analysis.py
    │   │
    │   └── utils/
    │         config.py
    │         logger.py
    │         helper.py
    │
    ├── models/
    │
    ├── results/
    │     metrics.csv
    │     predictions.csv
    │
    ├── figures/
    │
    ├── experiments/
    │     experiment_log.xlsx
    │
    ├── main.py
    │
    ├── requirements.txt
    │
    └── README.md
