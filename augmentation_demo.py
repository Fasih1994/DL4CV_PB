from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-o', '--output', required=True,
                help='path for output augmented images')
ap.add_argument('-p', '--prefix', type=str, default='image',
                help='prefix string to be attached before generated images')
args = vars(ap.parse_args())
# load image
print("[INFO] loading images...")
image = load_img(args['image'])
original = image.copy()
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct image data generator
augment = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, fill_mode='nearest',
                             horizontal_flip=True)
total = 0
print('[INFO] generating images...')
imageGen = augment.flow(image, batch_size=1, save_to_dir=args['output'],
                        save_prefix=args['prefix'], save_format='jpg')
for image in imageGen:
    total += 1
    if total == 10:
        break
