import argparse
import logging
import os
import multiprocessing as mp
from glob import glob

from pysot.core.config import cfg
from shell import DATASETS

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='union')
parser.add_argument('--config_files', type=str,
                    help='a string, if config file more than one, use comma to separate')
parser.add_argument("--base_config_path", type=str, default="../experiments",
                    help="the root path of config files")
parser.add_argument('--names', type=str,
                    help='dataset name for testing after training finished.'
                         'a string, if names more than one, use comma to separate. ')
parser.add_argument('--vis', action='store_true',
                    help='whether visualize result')
parser.add_argument('--device_ids', default='0,1', type=str,
                    help='a string, device used to training and testing, use comma to split more devices')
parser.add_argument('--num_proc', default=4, type=int,
                    help='number of processes for testing')
parser.add_argument('--start_epoch', default=15, type=int,
                    help='which epoch to start testing, this will test all the snapshot that >= start_epoch')
parser.add_argument('--port', default=2333, type=int,
                    help='pytorch distribute port')
parser.add_argument('--tracker_prefix', default='checkpoint_e', type=str,
                    help='all the results start with tracker_prefix')
parser.add_argument('--train_py', default='train_plus.py', type=str,
                    help='which file used for training')
parser.add_argument('--training', action='store_true',
                    help='whether start training')
parser.add_argument('--testing', action='store_true',
                    help='whether start testing')
parser.add_argument('--evaling', action='store_true',
                    help='whether start evaling')
args = parser.parse_args()


class Union(object):
    def __init__(self, config_files, base_config_path, names, vis, device_ids,
                 num_proc, start_epoch, port=2333, tracker_prefix="*", train_py="train_plus.py",
                 training=True, testing=True, evaling=False):
        datasets = [name.upper() for name in names]
        dataset_roots = [DATASETS[name] for name in datasets]

        self.config_files = ["%s/%s" % (base_config_path, config) for config in config_files]
        self.datasets = datasets
        self.dataset_roots = dataset_roots
        self.vis = vis
        self.device_ids = device_ids
        self.num_proc = num_proc
        self.start_epoch = start_epoch
        self.port = port
        self.tracker_prefix = tracker_prefix
        self.train_py = train_py
        self.training = training
        self.testing = testing
        self.evaling = evaling

    def __finish_training__(self, snapshot_dir):
        if os.path.exists(snapshot_dir):
            if len(os.listdir(snapshot_dir)) >= 19:
                return True
        return False

    def __train__(self, port, config, train_py):
        cmd = "python -m torch.distributed.launch " \
              "--nproc_per_node=2 --master_port=%d ../tools/%s --cfg %s" % (port, train_py, config)
        os.system(cmd)

    def run_train(self):
        for config in self.config_files:
            logger.info("start training %s" % config)
            print("start training %s" % config)
            cfg.merge_from_file(config)
            snapshot_dir = os.path.join(cfg.RESULTS_BASE_PATH, cfg.TRAIN.SNAPSHOT_DIR)
            if not self.__finish_training__(snapshot_dir):
                self.__train__(self.port, config, self.train_py)
                logger.info("finish training %s" % config)
                print("finish training %s" % config)
            else:
                logger.info("already training %s" % config)
                print("already training %s" % config)
            print()

    def __get_rest_snapshots__(self, snapshot_path, tracker_path, dataset, tracker_prefix="", start_epoch=0):
        snapshots = sorted(glob("%s/%s*" % (snapshot_path, tracker_prefix)),
                           key=lambda x: int(x.split("e")[-1].split(".")[0]))
        tested_snapshots = sorted(glob("%s/%s/*" % (tracker_path, dataset)),
                                  key=lambda x: int(x.split("e")[-1].split(".")[0]))
        snapshots = [x.replace("\\", "/") for x in snapshots]
        tested_snapshots = [x.replace("\\", "/") for x in tested_snapshots]
        tested_snapshots = [x.split("/")[-1] for x in tested_snapshots]
        rest_snapshots = []
        for snapshot in snapshots:
            snapshot_name = snapshot.split("/")[-1].split(".")[0]
            epoch = int(snapshot_name.split("e")[-1])
            if snapshot_name not in tested_snapshots and epoch >= start_epoch:
                rest_snapshots.append(snapshot)

        # rest_snapshots = [x for x in snapshots if x.split("/")[-1].split(".")[0] not in tested_snapshots]
        # rest_snapshots = [x for x in rest_snapshots if int(x.split("/")[-1].split(".")[0]) >= start_epoch]
        return rest_snapshots

    def __test__(self, dataset, dataset_root, config_file, snapshot, device_id, vis="--vis"):
        args = "--dataset %s --dataset_root %s --config %s --snapshot %s %s" % (
            dataset, dataset_root, config_file, snapshot, vis)
        cmd = "CUDA_VISIBLE_DEVICES=%s python ../tools/test.py %s" % (device_id, args)
        print(cmd)
        os.system(cmd)

    def run_test(self):
        pool = mp.Pool(processes=self.num_proc)  # 20
        device_ids = [int(idx) for idx in self.device_ids.split(",")]
        device_num = len(device_ids)
        datasets = self.datasets
        dataset_roots = self.dataset_roots
        count = 0  # control the device id

        for config in self.config_files:
            cfg.merge_from_file(config)
            snapshot_path = os.path.join(cfg.RESULTS_BASE_PATH, cfg.TRAIN.SNAPSHOT_DIR)
            tracker_path = os.path.join(cfg.RESULTS_BASE_PATH, cfg.TEST.TRACKER_DIR)

            for i in range(len(datasets)):
                dataset = datasets[i]
                dataset_root = dataset_roots[i]
                rest_snapshots = self.__get_rest_snapshots__(snapshot_path=snapshot_path,
                                                             tracker_path=tracker_path,
                                                             dataset=dataset,
                                                             tracker_prefix=self.tracker_prefix,
                                                             start_epoch=self.start_epoch)
                # print(rest_snapshots)

                for snapshot in rest_snapshots:
                    device_id = device_ids[count % device_num]
                    pool.apply_async(self.__test__, (dataset, dataset_root, config, snapshot, device_id, self.vis))
                    count += 1

        pool.close()
        pool.join()

    def __eval__(self, dataset, dataset_root, tracker_path, tracker_prefix):
        args = "--tracker_path %s --dataset %s --num 6 --dataset_root %s --tracker_prefix %s" % (
            tracker_path, dataset, dataset_root, tracker_prefix)
        cmd = "python ../tools/eval.py %s" % args
        print(cmd)
        os.system(cmd)

    def run_eval(self):
        datasets = self.datasets
        dataset_roots = self.dataset_roots
        tracker_prefix = self.tracker_prefix
        for i in range(len(datasets)):
            dataset = datasets[i]
            dataset_root = dataset_roots[i]
            for config in self.config_files:
                cfg.merge_from_file(config)
                tracker_path = os.path.join(cfg.RESULTS_BASE_PATH, cfg.TEST.TRACKER_DIR, dataset)
                print(config)
                self.__eval__(dataset, dataset_root, tracker_path, tracker_prefix)

    def __call__(self):
        if self.training:
            msg = "----------------------------------------------------------------\n" + \
                  "                         training                                \n" + \
                  "----------------------------------------------------------------"
            logger.info(msg)
            print(msg)
            self.run_train()

        if self.testing:
            msg = "----------------------------------------------------------------\n" + \
                  "                         testing                                \n" + \
                  "----------------------------------------------------------------"
            logger.info(msg)
            print(msg)
            self.run_test()

        if self.evaling:
            msg = "----------------------------------------------------------------\n" + \
                  "                         evaling                                \n" + \
                  "----------------------------------------------------------------"
            logger.info(msg)
            print(msg)
            self.run_eval()


if __name__ == '__main__':
    # config_files = args.config_files.split(",")
    # base_config_path = args.base_config_path
    # names = args.names.split(",")
    # vis = args.vis
    # device_ids = args.device_ids
    # num_proc = args.num_proc
    # start_epoch = args.start_epoch
    # port = args.port
    # tracker_prefix = args.tracker_prefix
    # train_py = args.train_py
    # training = args.training
    # testing = args.testing
    # evaling = args.evaling
    base_config_path = "../experiments"
    config_files = ["mybackbone_2gpu_5_union/5_union_res50_mod_layer3_reg_cross_rpn.yaml"]
    names = ["OTB100", "VOT2019", "UAV123"]
    vis = ""
    device_ids = "0"
    num_proc = 2
    start_epoch = 16
    port = 2333
    tracker_prefix = "ch*"
    train_py = "train_debug.py"
    training = True
    testing = True
    evaling = True

    union = Union(config_files=config_files,
                  base_config_path=base_config_path,
                  names=names,
                  vis=vis,
                  device_ids=device_ids,
                  num_proc=num_proc,
                  start_epoch=start_epoch,
                  port=port,
                  tracker_prefix=tracker_prefix,
                  train_py=train_py,
                  training=training,
                  testing=testing,
                  evaling=evaling)
    union()
