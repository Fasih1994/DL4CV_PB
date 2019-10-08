import cv2
import numpy as np
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        """

        :param preprocessors: preprocessors to be used
        """
        self.preprocessors = preprocessors

        #if preprocessors are none initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        #initialize the list of features and labels
        data = []
        lables = []

        #loop over input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            lable = imagePath.split(os.path.sep)[-2]

            #CHECK to see if preprocessors are not None
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            lables.append(lable)
            #show an update every verbose
            if verbose > 0 and i > 0 and (i+1) % verbose ==0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

        return np.array(data), np.array(lables)