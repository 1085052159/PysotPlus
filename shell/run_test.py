import os
import multiprocessing as mp
from glob import glob


def test(dataset, dataset_root, config_file, snapshot, vis="--vis"):
    args = "--dataset %s --dataset_root %s --config %s --snapshot %s %s" % (
        dataset, dataset_root, config_file, snapshot, vis)
    cmd = "python ../tools/test.py %s" % args
    print(cmd)
    os.system(cmd)


def test_with_multi_process(dataset, dataset_root, config_file, vis="--vis", snapshot_path="./snapshot",
                            tracker_path="./results", processes=4):
    def get_rest_snapshots():
        snapshots = sorted(glob("%s/*" % snapshot_path), key=lambda x: int(x.split("e")[-1].split(".")[0]))
        tested_snapshots = sorted(glob("%s/%s/*" % (tracker_path, dataset)), key=lambda x: int(x.split("e")[-1]))
        snapshots = [x.replace("\\", "/") for x in snapshots]
        tested_snapshots = [x.replace("\\", "/") for x in tested_snapshots]
        tested_snapshots = [x.split("/")[-1] for x in tested_snapshots]
        rest_snapshots = [x for x in snapshots if x.split("/")[-1].split(".")[0] not in tested_snapshots]
        return rest_snapshots

    rest_snapshots = get_rest_snapshots()
    pool = mp.Pool(processes=processes)
    for snapshot in rest_snapshots:
        pool.apply_async(test, (dataset, dataset_root, config_file, snapshot, vis))

    pool.close()
    pool.join()


def run_test():
    dataset = "OTB100"
    dataset_root = "%s/OTB2015" % base_path
    snapshot = "../experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth"
    config_file = "../experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml"

    # snapshot = "../experiments/siamrpn_alex_dwxcorr_otb/model.pth"
    # config_file = "../experiments/siamrpn_alex_dwxcorr_otb/config.yaml"

    # dataset = "VOT2018"
    # dataset_root = "%s/VOT2018/sequences" % base_path
    # dataset = "VOT2016"
    # dataset_root = "%s/VOT2016" % base_path
    # dataset = "VOT2019"
    # dataset_root = "%s/VOT2019" % base_path
    # dataset = "GOT-10k"
    # dataset_root = "%s/GOT-10k" % base_path
    # dataset = "UAV123"
    # dataset = "UAV20L"
    # dataset_root = "%s/UAV123" % base_path
    # dataset = "LaSOT"
    # dataset_root = "%s/LaSOT" % base_path
    # snapshot = "../experiments/siamrpn_r50_l234_dwxcorr/model.pth"
    # config_file = "../experiments/siamrpn_r50_l234_dwxcorr/config.yaml"
    # snapshot = "../experiments/siamrpn_alex_dwxcorr/model.pth"
    # config_file = "../experiments/siamrpn_alex_dwxcorr/config.yaml"

    # snapshot = "../experiments/mine/5_batch_32_e7.pth"
    # config_file = "../experiments/mine/VOT2016.yaml"
    # snapshot = "../experiments/mine/5_batch_32_e19.pth"
    # snapshot = "../experiments/mine/5_batch_128_e18.pth"
    # snapshot = "../experiments/mine/5_batch_32_cbam_e19.pth"
    # config_file = "../experiments/mine/5_batch_32_cbam_e19.yaml"
    # config_file = "../experiments/mine/VOT2018_e19_0.410.yaml"
    # config_file = "../experiments/mine/VOT2018_e19_0.412.yaml"
    # config_file = "../experiments/mine/VOT2019_e19_0.282.yaml"
    # config_file = "../experiments/mine/VOT2019_e19_0.284.yaml"
    # snapshot = "../experiments/mybackbone_2gpu_5_union/checkpoint_e12.pth"
    # config_file = "../experiments/mybackbone_2gpu_5_union/5_union_best_batch_32_mybackbone.yaml"
    # snapshot = "C:/Users/YSQPCL/Desktop/snapshots_res50_mod_layer3_connect1_feature_confusion_rpn_cls_sum/checkpoint_e19.pth"
    # config_file = "../experiments/debug.yaml"

    vis = "--vis "
    # vis = ""
    test(dataset=dataset,
         dataset_root=dataset_root,
         config_file=config_file,
         snapshot=snapshot,
         vis=vis)


def run_test_with_multi_processes():
    # dataset = "OTB100"
    # dataset_root = "%s/OTB2015" % base_path
    # dataset = "VOT2018"
    # dataset_root = "%s/VOT2018/sequences" % base_path
    dataset = "VOT2016"
    dataset_root = "%s/VOT2016" % base_path
    config_file = "../experiments/siamrpn_r50_l234_dwxcorr_2gpu_6_union/6_union_best_batch_32.yaml"
    vis = "--vis "
    vis = ""

    snapshot_path = "./snapshot_32"
    tracker_path = "./results_32"
    processes = 2
    test_with_multi_process(dataset=dataset,
                            dataset_root=dataset_root,
                            config_file=config_file,
                            vis=vis,
                            snapshot_path=snapshot_path,
                            tracker_path=tracker_path,
                            processes=processes)


if __name__ == '__main__':
    base_path = "/media/ubuntu/dataset_nvme/object_tracking"
    base_path = "F:/dataset"
    run_test()
    # run_test_with_multi_processes()
