dataset='cifar10'
seeds=$(echo {0..9})
bs=128
cuda=1
lr='1e-3'
epoch='0 1 2 5 10 15 20'

# these variables usually do not need to change along with the dataset.
nets=('resnet18' 'resnet34' 'resnet50')
order=2
n_jobs=1
fc_name='fc'

for net in ${nets[@]}; do
    nohup python -u ./scripts/component-scores/grand.py \
    --dataset $dataset \
    --net $net \
    --epoch $epoch \
    --lr $lr \
    --bs $bs \
    --order $order \
    --fc_name $fc_name \
    --seed $seeds \
    --cuda $cuda \
    2>&1 >/dev/null
done
