# out/test_sp500_2020_0/csv_zoo_final.csv
CUDA_VISIBLE_DEVICES=3 python run_adaptive_combination.py \
    --expressions_file data/gfn_logs/pool_50/gfn_gnn_csi300_50_0-0.01-1.0-1.0-1.0-0.3-linear-0.0/pool_9999.json \
    --instruments csi300 \
    --threshold_ric 0.015 \
    --threshold_ricir 0.15 \
    --chunk_size 400 \
    --window inf \
    --n_factors 50 \
    --cuda 2 \
    --train_end_year 2020 \
    --seed 0 \
    #--use_weights True