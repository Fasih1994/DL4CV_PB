# import packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.dataset import SimpleDatasetLoader
from pyimagesearch.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input
from imutils import paths
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
ap.add_argument('-o', '--output', required=True,
                help='path output model')
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode='nearest')

print('[INFO] loading data...')
imagePaths = list(paths.list_images(args['dataset']))
classnames = os.listdir(args['dataset'])
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
lds = SimpleDatasetLoader([aap, iap])
(data, labels) = lds.load(imagePaths, verbose=500)

data = data.astype('float') / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print('[INFO] building base model...')
baseModel = VGG16(weights='imagenet', include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

print('[INFO] building head model...')
headModel = FCHeadNet.build(baseModel=baseModel, classes=len(classnames), D=256)

print('[INFO] performing network surgery...')
model = Model(inputs= baseModel.input, outputs=headModel)

print('[INFO] freezing base model...')
for layer in baseModel.layers:
    layer.trainable = False

print('[INFO] compiling model...')
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print('[INFO] training model...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=25,
                    steps_per_epoch=len(trainX) // 25, verbose=1)

print('[INFO] evaluating after initialization... ')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classnames))

print('[INFO] unfreezing layer from layer # 15...')
for layer in baseModel.layers[15:]:
    layer.trainable = True

print('[INFO] re-compiling model...')
opt = SGD(lr=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('[INFO] fine-tuning model after warm-up...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=100,
                    steps_per_epoch=len(trainX)//32,verbose=1)

print('[INFO] evaluating after fine-tuning... ')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classnames))

print('[INFO] serializing model...')
model.save(args['output'])