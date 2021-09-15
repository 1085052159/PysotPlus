import os
import multiprocessing as mp

from shell import DATASETS


def tune(dataset, dataset_root, config, snapshot, gpu_id, trial):
    # args = "--dataset %s --dataset_root %s --config %s --snapshot %s --gpu_id %s" % (
    #     dataset, dataset_root, config, snapshot, gpu_id)
    args = "--dataset %s --dataset_root %s --config %s --snapshot %s --trial %d" % (
        dataset, dataset_root, config, snapshot, trial)
    cmd = "CUDA_VISIBLE_DEVICES=%s python ../tools/tune.py %s" % (gpu_id, args)
    print(cmd)
    os.system(cmd)


def run_tune():
    tune(dataset=dataset,
         dataset_root=dataset_root,
         config=config,
         snapshot=snapshot,
         gpu_id="0",
         trial=trial)


def run_tune_multi(num):
    pool = mp.Pool(processes=num)
    for i in range(num):
        # gpu_id = i % 2
        gpu_id = 1
        pool.apply_async(tune, (dataset, dataset_root, config, snapshot, gpu_id, trial))
    pool.close()
    pool.join()


if __name__ == '__main__':
    base_path = "/media/ubuntu/dataset_nvme/object_tracking"
    # base_path = "/media/linux/dataset_nvme/dataset"
    dataset = "OTB100"
    dataset = "VOT2018"
    # dataset = "VOT2016"
    # dataset = "VOT2019"
    dataset = "UAV123"
    dataset = "LaSOT"
    dataset_root = DATASETS[dataset]

    config = "../experiments/siamrpn_r50_l234_dwxcorr_2gpu_5_union/5_union_best_batch_32.yaml"
    # snapshot = "/home/ubuntu/PycharmProjects/Results/snapshots/5/snapshot_best_batch_32/checkpoint_e7.pth"
    snapshot = "/home/ubuntu/PycharmProjects/Results/snapshots/5/snapshot_best_batch_32/checkpoint_e19.pth"
    trial = 1
    # run_tune()
    run_tune_multi(1)