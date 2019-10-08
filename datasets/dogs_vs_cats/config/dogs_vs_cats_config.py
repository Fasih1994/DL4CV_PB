# define path to image directory
IMAGES_PATH = "../../kaggle_dogs_vs_cats/train"

# we do not have testing and validation data
# so make it from training data :)
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES


# defin path to output validation, training and testing
# HDF5 files
TRAIN_HDF5 = "../../kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "../../kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../../kaggle_dogs_vs_cats/hdf5/test.hdf5"

# define path to output model
MODEL_PATH = "../output/alexnet_dogs_vs_cats.model"

# define output path for mean
DATASET_MEAN = "../output/dogs_vs_cats_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"