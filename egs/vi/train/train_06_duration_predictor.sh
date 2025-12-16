CUDA_VISIBLE_DEVICES=1 python src/f5_tts/train/train_duration_predictor.py \
    --train_path data/vi/train_with_alignments.data \
    --test_path data/vi/test_with_alignments.data \
    --vocab_path ckpts/f5tts_vi_v1_base/vocab.txt \
    \
    --text_dim 512 \
    --filter_channels 32 \
    --kernel_size 3 \
    --dropout 0.5 \
    \
    --batch_size_frames 500000 \
    --max_samples 10000 \
    --num_epoch 20 \
    --learning_rate 1e-4 \
    --num_warmup_steps 1000 \
    --grad_clip_norm 1.0 \
    \
    --save_dir ckpts/dp \
    --save_per_steps 100000000 \
    --log_per_steps 10