dataset='caltech256'

# nohup python -u scripts/metadata/aum.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
# nohup python -u scripts/metadata/centroid.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
nohup python -u scripts/metadata/consistence.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
nohup python -u scripts/metadata/el2n.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
wait
nohup python -u scripts/metadata/forgetting.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
nohup python -u scripts/metadata/grand.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
wait
nohup python -u scripts/metadata/isolation-forest.py --dataset $dataset --cuda 0 2>&1 >/dev/null &
wait
nohup python -u scripts/metadata/oc-svm.py --dataset caltech101 --cuda 0 2>&1 >/dev/null &
wait
nohup python -u scripts/metadata/collection.py --dataset $dataset 2>&1 >/dev/null