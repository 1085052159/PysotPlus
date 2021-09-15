import os

from shell import DATASETS


def eval(dataset, dataset_root, tracker_path, tracker_prefix="ch", show_video_level=False, vis=False):
    if show_video_level:
        show_video_level = "--show_video_level "
    else:
        show_video_level = ""
    if vis:
        vis = "--vis"
    else:
        vis = ""
    args = "--tracker_path %s --dataset %s --num 3 --dataset_root %s --tracker_prefix %s %s %s" % (
        tracker_path, dataset, dataset_root, tracker_prefix, show_video_level, vis)
    cmd = "python ../tools/eval.py %s" % args
    print(cmd)
    os.system(cmd)


def run_eval():
    dataset = "OTB100"
    dataset_root = DATASETS[dataset]
    tracker_path = "paper02/results/5/results_debug/%s" % dataset
    tracker_prefix = "ch*"
    show_video_level = False
    vis = True

    eval(dataset=dataset,
         dataset_root=dataset_root,
         tracker_path=tracker_path,
         tracker_prefix=tracker_prefix,
         show_video_level=show_video_level,
         vis=vis)


if __name__ == '__main__':
    run_eval()
