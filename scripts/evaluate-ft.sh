nohup python -u ./scripts/evaluate-ft.py \
--dataset cifar10 \
--collection_date '230827-0851' \
--k 10 \
--max_iter 500 \
--n_eval 100 \
--eta 0.9 \
--alpha 0.9 \
--beta 0.99 \
--eval_n_jobs 10 \
--embed_net 'vit' \
--embed_dataset 'imagenet21k' \
--embed_epoch -1 \
--embed_date '666666-6666' \
--epoch 10 \
--lr '1e-3' \
--bs 128 \
--gamma 0.95 \
--seed 0 \
--cuda 0 \
2>&1 >./evaluate-ft-cifar10.log &


# nohup python -u ./scripts/evaluate-ft.py \
# --dataset cifar100 \
# --collection_date '230827-0851' \
# --k 10 \
# --max_iter 500 \
# --n_eval 200 \
# --eta 0.9 \
# --alpha 0.9 \
# --beta 0.99 \
# --eval_n_jobs 10 \
# --embed_net 'vit' \
# --embed_dataset 'imagenet21k' \
# --embed_epoch -1 \
# --embed_date '666666-6666' \
# --epoch 10 \
# --lr '1e-2' \
# --bs 128 \
# --gamma 0.95 \
# --seed 0 \
# --cuda 1 \
# 2>&1 >./evaluate-ft-cifar100.log &
