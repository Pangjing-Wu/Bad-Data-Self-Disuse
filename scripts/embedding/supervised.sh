nohup \
python -u ./scripts/embedding/supervised.py \
--dataset cifar10 \
--net resnet18 \
--bs 128 \
--lr 1e-3 \
--ckp 10 \
--epoch 50 \
--augment \
2>&1 >/dev/null &

nohup \
python -u ./scripts/embedding/supervised.py \
--dataset cifar100 \
--net resnet18 \
--bs 128 \
--lr 1e-3 \
--ckp 20 \
--epoch 200 \
--augment \
2>&1 >/dev/null &