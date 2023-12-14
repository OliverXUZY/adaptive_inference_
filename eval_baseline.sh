# python tools/eval_baselines.py --model vit


# python tools/eval_baseline.py \
#     --model vit \
#     --skip_block 0 \
#     --log_path debug \
#     --limit 10

# python tools/eval_baseline.py \
#     --model resnet18 \
#     --limit 5000 \
#     --dataset imagenet \
#     --skip_block 1

for skip_block in {0..7}
do
    python tools/eval_baseline.py \
        --model resnet18 \
        --limit 5000 \
        --dataset imagenet \
        --skip_block $skip_block
done
