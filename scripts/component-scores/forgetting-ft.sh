dataset='cifar10'
seeds=$(echo {0..9})
bs=128
cuda=0
lr='1e-2'
epoch=50
gamma=0.95

# these variables usually do not need to change along with the dataset.
embed_nets=('vit' 'mae' 'beit')
embed_dataset='imagenet21k'
embed_date='666666-6666'
n_jobs=1

for net in ${embed_nets[@]}; do
    nohup python -u ./scripts/component-scores/forgetting-ft.py \
    --dataset $dataset \
    --embed_net $net \
    --embed_dataset $embed_dataset \
    --embed_date $embed_date \
    --epoch $epoch \
    --lr $lr \
    --bs $bs \
    --gamma $gamma \
    --seed $seeds \
    --cuda $cuda \
    2>&1 >/dev/null
done