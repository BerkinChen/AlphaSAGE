python run_adaptive_combination.py \
    --expressions_file data/gfn_logs/pool_50/gfn_gnn_csi300_50_0-0.0-1.0-1.0-0.5-0.7-linear-0.0/pool_9999.json \
    --chunk_size 400 \
    --window inf \
    --cuda 0 \
    --n_factors 20 \
    --train_end_year 2020 \
    --seed 0