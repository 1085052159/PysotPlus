import os
import warnings

warnings.filterwarnings("ignore")


def run_train():
    config_file = "../experiments/mybackbone_2gpu_5_union/5_union_res50_mod_layer3_reg_feature_confusion.yaml"

    cmd = "python -m torch.distributed.launch " \
          "--nproc_per_node=1 --master_port=2333 ../tools/train.py --cfg %s" % config_file
    print(cmd)
    os.system(cmd)


def run_train_half():
    path = "../experiments/siamrpn_r50_l234_dwxcorr"
    configs = ["5_union_rpnpp_l234.yaml"]

    for config in configs:
        config_file = "%s/%s" % (path, config)

        cmd = "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch " \
              "--nproc_per_node=2 --master_port=2334 ../tools/train_plus.py --cfg %s" % config_file
        os.system(cmd)


def run_train_single():
    path = "../experiments/siamrpn_r50_l234_dwxcorr"
    configs = ["5_union_rpnpp_l234.yaml"]

    for config in configs:
        config_file = "%s/%s" % (path, config)

        cmd = "python ../tools/train_half_single_gpu.py --cfg %s" % config_file
        os.system(cmd)

    os.system(cmd)


if __name__ == '__main__':
    # run_train()
    # run_train_half()
    run_train_single()
