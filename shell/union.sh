echo "init start"
cd /home/ubuntu/PycharmProjects/PysotPlus/shell
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
export PYTHONPATH=/home/ubuntu/PycharmProjects/PysotPlus:$PYTHONPATH
echo "init done"

# the root path of configs
base_config_path="../experiments"
# train siamrpnpp and siamban sequentially
# if you have more than one configs, please use "," to seperate them, just like the following
config_files="siamrpn_r50_l234_dwxcorr/5_union_rpnpp_l234.yaml,siamban_r50_l234_dwxcorr/5_union_ban_l234.yaml"
names="OTB100,VOT2019,UAV123"
device_ids="0,1"
num_proc=2 # equal to the num of gpus
start_epoch=16 # the epoch to start testing, this means that when finishing training, 16-end_epoch snapshots will test
port=2333
tracker_prefix="ch*"
train_py="train_half.py"
training="--training"
testing="--testing"
evaling="--evaling"
vis=""
#vis="--vis"
#training="" # if no training, please uncomment this
#testing="" # if no testing, please uncomment this
#evaling="" # if no evaling, please uncomment this
python union.py --config_files $config_files --base_config_path $base_config_path \
--names $names $vis --device_ids $device_ids --num_proc $num_proc \
--start_epoch $start_epoch --port $port --tracker_prefix $tracker_prefix \
--train_py $train_py $training $testing $evaling
