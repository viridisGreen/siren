clear


python main.py \
    --batch_size=1 \
    --experiment_name=test_$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S) \
    --num_epochs=100 \
    --epochs_til_ckpt=1 \
    \
    --image_path=./data/test_001.tif
