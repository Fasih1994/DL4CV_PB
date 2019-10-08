import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype('float'))
        R -= self.rMean
        B -= self.bMean
        G -= self.gMean
        return cv2.merge([B, G, R])
