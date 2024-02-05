dataset='cifar10'
bs=512
cuda=0
pca_dims=(10 25 50 100)
n_estimators=500
svm_kernel='rbf'
rbf_gamma='scale'

# these variables usually do not need to change along with the dataset.
embed_nets=('vit' 'mae' 'beit' 'resnet18' 'resnet34' 'resnet50')
embed_datasets=('imagenet21k' 'imagenet21k' 'imagenet21k' 'imagenet1k' 'imagenet1k' 'imagenet1k')
embed_algos=('pt' 'pt' 'pt' 'pt' 'pt' 'pt')
embed_dates=('666666-6666' '666666-6666' '666666-6666' '666666-6666' '666666-6666' '666666-6666')
embed_epoch=-1

for pca_dim in ${pca_dims[@]}; do
    for i in $(seq 0 $[ ${#embed_nets[*]} - 1 ]); do
        nohup python -u ./scripts/component-scores/one-class-svm.py \
        --dataset $dataset \
        --embed_net ${embed_nets[i]} \
        --embed_dataset ${embed_datasets[i]} \
        --embed_algo ${embed_algos[i]}\
        --embed_epoch $embed_epoch \
        --embed_date ${embed_dates[i]} \
        --pca_dim $pca_dim \
        --bs $bs \
        --svm_kernel $svm_kernel \
        --rbf_gamma $rbf_gamma\
        --cuda $cuda \
        2>&1 >/dev/null
    done
done