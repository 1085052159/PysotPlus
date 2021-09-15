echo "init start"
cd /home/ubuntu/PycharmProjects/PysotPlus/shell
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
export PYTHONPATH=/home/ubuntu/PycharmProjects/PysotPlus:$PYTHONPATH
echo "init done"

declare -A DATASETS
DATASETS["OTB100"]="/media/ubuntu/dataset_nvme/object_tracking/OTB2015"
DATASETS["VOT2016"]="/media/ubuntu/dataset_nvme/object_tracking/VOT2016"
DATASETS["VOT2018"]="/media/ubuntu/dataset_nvme/object_tracking/VOT2018/sequences"
DATASETS["VOT2019"]="/media/ubuntu/dataset_nvme/object_tracking/VOT2019"
DATASETS["VOT2020"]=""
DATASETS["GOT-10k"]="/media/ubuntu/dataset_nvme/object_tracking/GOT-10k/test"
DATASETS["LaSOT"]="/media/ubuntu/dataset_nvme/object_tracking/LaSOT"
DATASETS["UAV123"]="/media/ubuntu/dataset_nvme/object_tracking/UAV123"
DATASETS["UAV20L"]="/media/ubuntu/dataset_nvme/object_tracking/UAV123"
dataset="OTB100"
dataset="VOT2016"
# dataset="VOT2018"
# dataset="VOT2019"
# dataset="GOT-10k"
dataset="LaSOT"
# dataset="UAV123"
# dataset="UAV20L"
dataset_root=${DATASETS[$dataset]}
echo "dataset_root: $dataset_root"

cfg_partial="rpnpp_l234"
cfg_name="../experiments/siamrpn_r50_l234_dwxcorr/5_union_${cfg_partial}.yaml"

root_path="/home/ubuntu/PycharmProjects/Results/PysotPlus"
ckpt="checkpoint_e19"
snapshot="${root_path}/snapshots/5/snapshots_${cfg_partial}/${ckpt}.pth"

echo "snapshot: $snapshot"
echo "cfg_name: $cfg_name"
trial=100
device_id=0
CUDA_VISIBLE_DEVICES=$device_id python ../tools/tune.py --dataset $dataset --dataset_root $dataset_root --config $cfg_name --snapshot $snapshot --trial $trial
