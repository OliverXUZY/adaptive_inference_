# python train.py \
#     -c '/srv/home/zxu444/vision/adaptive_inference/configs/resnet18_cifar10.yaml' \
#     -n 'train_resnet18_cifar10' \
#     -pf 1


# Start GPU monitoring in the background
# (
#     while true; do
#         nvidia-smi | tee -a ./log/gpu_usage_${SLURM_JOB_ID}.log
#         sleep 60  # Log every 60 seconds
#     done
# ) &
# monitor_pid=$!


python train.py \
    -c '/srv/home/zxu444/vision/adaptive_inference/configs/resnet50_imagenet.yaml' \
    -n 'train_resnet50_imagenet_debug' \
    -pf 1
