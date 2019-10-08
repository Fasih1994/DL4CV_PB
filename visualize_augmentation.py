from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import argparse
import os


# define a visualization function
def show_with_plt(image, title, pos):
    plt.subplot(3, 5, pos)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='path to augmented images')
args = vars(ap.parse_args())

plt.suptitle('Augmentation example')
show_with_plt(load_img('beagle.png'), 'original', 3)
for (i, image) in enumerate(os.listdir(args['input'])):
    show_with_plt(load_img(os.path.join(args['input'], image)),
                  "gen_" + str(i + 1), i + 6)
plt.show()
