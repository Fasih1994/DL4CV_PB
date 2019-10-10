# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, horizontal_flip=True, shear_range=0.15,
                         fill_mode='nearest')
# load RGB means from training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()


# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
                                prerprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128,
                              prerprocessors=[pp, mp, iap], classes=2)

print("[INFO] Compiling model....")
opt = Adam(lr=1e-3)
model = AlexNet.build(227, 227, 3,
                      classes=2, reg=0.0002)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['acc'])

path = os.path.sep.join([config.OUTPUT_PATH, '{}.png'.format(os.getpid())])
callbacks = [TrainingMonitor(path)]


print("[INFO] training the network...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch= trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=75,
    max_queue_size=30,
    callbacks=callbacks, verbose=1)

print('[INFO] serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()