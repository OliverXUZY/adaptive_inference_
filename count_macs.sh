# python count_macs.py --arch resnet18 \
#         --dataset cifar10 \
#         --path /Users/zyxu/Documents/py/vision/adaptive_inference/test

python count_macs.py --arch resnet50 \
        --dataset imagenet \
        --path libs/model/macs/timm | tee debug/output_ResNet.txt
