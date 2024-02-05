dataset='cifar10'
seeds=$(echo {0..9})
bs=128
cuda=0
lr='1e-3'
epoch=100

# these variables usually do not need to change along with the dataset.
nets=('resnet18' 'resnet34' 'resnet50')
n_jobs=1

for net in ${nets[@]}; do
    nohup python -u ./scripts/component-scores/forgetting.py \
    --dataset $dataset \
    --net $net \
    --epoch $epoch \
    --lr $lr \
    --bs $bs \
    --seed $seeds \
    --cuda $cuda \
    2>&1 >/dev/null
done
