import cv2
import numpy as np

class CropPreprocessor:
    def __init__(self, width, height, horizental_flip=True, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        self.horizental_flip = horizental_flip

    def preprocess(self, image):
        crops = []

        (h, w) = image.shape[:2]

        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h-self.height, h, w],
            [0, h - self.height, self.width, w]
        ]
        dw = int(0.5 * (w - self.width))
        dh = int(0.5 * (h - self.height))

        coords.append([dw, dh, w-dw, h-dh])

        for (startX, startY, endX, endY) in coords:
            crop = image[startY: endY, startX: endX]
            crop = cv2.resize(crop, (self.width, self.height),
                              interpolation= self.inter)
            crops.append(crop)
        if self.horizental_flip:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)