import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """
        store the target image width, height and interpolation
        :param width:
        :param height:
        :param inter: cv2.Interpolation
        """
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self,image):
        """
        resize image ignoring the aspect ratio
        :param image: image to be resized
        :return: resized Image
        """
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)
