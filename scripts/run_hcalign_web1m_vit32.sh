# ViT-B/32
DATA_PATH="data/WebVid/"
VIDEO_DIR='data/WebVid/videos/'
OUT_DIR=outputs/webvid_pretrain/train1m
SEED=42 
BS=512

CUDA_VISIBLE_DEVICES=4,5,6,7  taskset -c 40-80  \
python -m torch.distributed.launch --nproc_per_node=4 --master_port 54 \
    main_hcalign_pretrain.py  --num_thread_reader=4 --do_train  \
    --lr 1e-3 --batch_size=${BS}  --batch_size_val 48 \
    --epochs=5  --n_display=20 \
    --data_path ${DATA_PATH} \
    --train_csv 'train_1M.json' \
    --features_path $VIDEO_DIR \
    --output_dir ${OUT_DIR} --seed 42 \
    --max_words 32 --max_frames 4    --datatype webvid  \
    --feature_framerate 1 --coef_lr 1e-4 \
    --freeze_layer_num 0  --btm_mode all --slice_framepos 2  \
    --linear_patch 2d --sim_header seqTransf  --pretrained_clip_name ViT-B/32 \
    --cc_msl_weight 0.5 --cc_start_epoch 0 --cc_selection_thresh 0.5 --cc_sample_frames 2

