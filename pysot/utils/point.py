import cv2
import numpy as np


class Point:
    """
    This class generate points.
    """

    def __init__(self, stride, size, image_center):
        self.stride = stride
        self.size = size
        self.image_center = image_center

        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)

        return points


# if __name__ == '__main__':
#     p = Point(stride=8, size=25, image_center=255 // 2)
#     print(p.points[0][0])
#     print(p.points[1][0])
#     print(p.points[0][0])
#     img = np.ones((255, 255, 3), dtype=np.float32)
#     cv2.circle(img, (p.points[0][0][0], p.points[1][0][0]), 1, (0, 0, 255), 3, 3)
#     cv2.ellipse(img, (150, 150), (25, 25), 0, 0, 360, color=(255, 0, 0), thickness=3)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
#     print(p.points.shape)
