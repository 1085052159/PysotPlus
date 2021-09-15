import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import corner2center
from pysot.utils.point import Point


class PointTarget:
    def __init__(self, ):
        self.points = Point(cfg.ANCHOR.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def __call__(self, target, size, neg=False):

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)
        delta_weight = np.zeros((size, size), dtype=np.float32)
        overlap = np.zeros((size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        if neg:
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0

            return cls, delta, delta_weight, overlap

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                       np.square(tcy - points[1]) / np.square(th / 4) < 1)
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / 2) +
                       np.square(tcy - points[1]) / np.square(th / 2) > 1)

        # sampling
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1
        cls[neg] = 0

        return cls, delta, delta_weight, overlap


if __name__ == '__main__':
    pt = PointTarget()
    pt([100, 100, 200, 200], 25)