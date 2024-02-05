nohup python -u ./scripts/evaluate.py \
--dataset cifar10 \
--collection_date '230821-0848' \
--k 10 \
--max_iter 50 \
--n_eval 100 \
--eta 0.9 \
--alpha 0.9 \
--beta 0.99 \
--eval_n_jobs 6 \
--net resnet18 \
--epoch 10 \
--lr '1e-3' \
--bs 512 \
--seed 0 \
--cuda 1 \
2>&1 >./evaluate.log &


# nohup python -u ./scripts/evaluate.py \
# --dataset cifar10 \
# --collection_date '230821-0848' \
# --k 3 \
# --max_iter 50 \
# --n_eval 10 \
# --eta 1.0 \
# --alpha 0.9 \
# --beta 1.0 \
# --eval_n_jobs 4 \
# --net resnet18 \
# --epoch 1 \
# --lr '1e-3' \
# --bs 128 \
# --seed 0 \
# --cuda 0 \
# 2>&1 >./evaluate-temp.log &