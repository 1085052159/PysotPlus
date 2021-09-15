export PYTHONPATH=/home/linux/PycharmProjects/PysotPlus:$PYTHONPATH

dataset="OTB100"
dataset_root="/media/linux/dataset_nvme/dataset/OTB2015"
tracker_path="./results"
tracker_prefix="checkpoint_e"

python ../tools/eval.py --tracker_path $tracker_path --dataset $dataset --num 6 --dataset_root $dataset_root --tracker_prefix $tracker_prefix