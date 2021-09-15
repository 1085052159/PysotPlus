export PYTHONPATH=/home/ubuntu/PycharmProjects/PysotPlus:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1
cfg_name="../experiments/siamrpn_r50_l234_dwxcorr_2gpu/config.yaml"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2333 ../tools/train.py --cfg $cfg_name
