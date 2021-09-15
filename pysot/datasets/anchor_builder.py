from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.point_target import PointTarget

ANCHORS = {
    'AnchorTarget': AnchorTarget,
    'PointTarget': PointTarget,
}


def build_target(anchor_type="AnchorTarget"):
    return ANCHORS[anchor_type]()


# if __name__ == '__main__':
#     anchor = build_target("AnchorTarget")
#     a, b, c, d = anchor([100, 100, 200, 200], 25, 0.2)
#     print(a.shape if a is not None else None)
#     print(b.shape if b is not None else None)
#     print(c.shape if c is not None else None)
#     print(d.shape if d is not None else None)
