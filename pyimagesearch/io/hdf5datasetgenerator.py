from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:

    def __init__(self, dbpath, batch_size, aug=None, binarize=True,
                 prerprocessors = None, classes = 2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batch_size = batch_size
        self.aug = aug
        self.preprocessors = prerprocessors
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbpath)
        self.numImages = self.db['labels'].shape[0]

    def generator(self, passes = np.inf):
        # initialize the epoch count
        epochs = 0

        # loops  infinitely
        while epochs < passes:

            # loops over images
            for i in np.arange(0, self.numImages, self.batch_size):
                images = self.db['images'][i: i + self.batch_size]
                labels = self.db['labels'][i: i + self.batch_size]

                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        procImages.append(image)
                    images = np.array(procImages)
                #  if data augmenter exist, apply it
                if self.aug is not None:
                    (images, labels) = next(
                        self.aug.flow(images, labels, batch_size= self.batch_size)
                    )
                yield (images, labels)

                epochs +=1

    def close(self):
        self.db.close()