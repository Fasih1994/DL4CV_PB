from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split('.')[0] for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, stratify=trainLabels,
                         test_size=config.NUM_TEST_IMAGES, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

split = train_test_split(trainPaths, trainLabels, stratify=trainLabels,
                         test_size=config.NUM_VAL_IMAGES, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
dataset = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ('test', testPaths, testLabels, config.TEST_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5)
]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

for (dtype, paths, labels, outputPath) in dataset:
    print('[INFO] building {}...'.format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath=outputPath)
    widgets = ['Building Dataset: ',progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pgbar = progressbar.ProgressBar(max_value=len(paths), widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dtype == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            B.append(b)
            G.append(g)
        writer.add([image], [label])
        pgbar.update(i)
    pgbar.finish()
    writer.close()

print('[INFO] serializing mean...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()