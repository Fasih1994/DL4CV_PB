from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.dataset import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import random
import os
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help='path to input model')
ap.add_argument('-d', '--dataset', required=True,
                help='path to dataset')
args = vars(ap.parse_args())

model = load_model(args['model'])
imagepaths = list(paths.list_images(args['dataset']))
classNames = sorted(os.listdir(args['dataset']))
random.shuffle(imagepaths)
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
imagePath = random.choice(imagepaths)
while True:

    image = cv2.imread(imagePath)
    original = image.copy()
    image = aap.preprocess(img_to_array(image))
    image = iap.preprocess(image)
    image = np.expand_dims(image, axis=0)

    label = imagePath.split(os.path.sep)[-2]
    pred = model.predict(image, batch_size=1)
    pred = classNames[pred.argmax(axis=1)[0]]
    cv2.putText(original, "label: {}".format(label, pred),
                (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (250, 250, 250), 2)
    cv2.imshow('Flower 17', original)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord('n'):
        cv2.putText(original, "predicted: {}".format(pred),
                    (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Flower 17', original)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        else:
            imagePath = random.choice(imagepaths)

cv2.destroyAllWindows()
