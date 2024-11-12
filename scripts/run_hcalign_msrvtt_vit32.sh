# ViT-B/32
DATA_PATH="data/MSRVTT"
VIDEO_DIR=data/MSR-VTT/videos_compressed/
OUT_DIR=outputs/msrvtt_train9k
SEED=42 
BS=256


CUDA_VISIBLE_DEVICES=4,5,6,7  taskset -c 40-80  \
python -m torch.distributed.launch --nproc_per_node=4 --master_port 53 \
    main_hcalign.py  --num_thread_reader=8 --do_train  \
    --lr 1e-3 --batch_size=${BS}  --batch_size_val 48 \
    --epochs=5  --n_display=10 \
    --train_csv ${DATA_PATH}//MSRVTT_train.9k.csv \
    --val_csv ${DATA_PATH}//MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT_data.json \
    --features_path $VIDEO_DIR \
    --output_dir ${OUT_DIR} --seed 42 \
    --max_words 32 --max_frames 12    --datatype msrvtt  \
    --feature_framerate 1 --coef_lr 1e-4 \
    --freeze_layer_num 0  --btm_mode all --slice_framepos 2  \
    --linear_patch 2d --sim_header seqTransf  --pretrained_clip_name ViT-B/32 \
    --cc_msl_weight 0.5 --cc_start_epoch 1 --cc_selection_thresh 0.5


