dataset='cifar10'
seeds=$(echo {0..9})
bs=512
cuda=0
pca_dims=(10 25 50 100)
n_estimators=500

# these variables usually do not need to change along with the dataset.
embed_nets=('vit' 'mae' 'beit' 'resnet18' 'resnet34' 'resnet50')
embed_datasets=('imagenet21k' 'imagenet21k' 'imagenet21k' 'imagenet1k' 'imagenet1k' 'imagenet1k')
embed_algos=('pt' 'pt' 'pt' 'pt' 'pt' 'pt')
embed_dates=('666666-6666' '666666-6666' '666666-6666' '666666-6666' '666666-6666' '666666-6666')
embed_epoch=-1
model_n_jobs=20

for pca_dim in ${pca_dims[@]}; do
    for i in $(seq 0 $[ ${#embed_nets[*]} - 1 ]); do
        nohup python -u ./scripts/component-scores/isolation-forest.py \
        --dataset $dataset \
        --embed_net ${embed_nets[i]} \
        --embed_dataset ${embed_datasets[i]} \
        --embed_algo ${embed_algos[i]}\
        --embed_epoch $embed_epoch \
        --embed_date ${embed_dates[i]} \
        --pca_dim $pca_dim \
        --bs $bs \
        --model_n_jobs $model_n_jobs \
        --n_estimators $n_estimators\
        --seed $seeds \
        --cuda $cuda \
        2>&1 >/dev/null
    done
done