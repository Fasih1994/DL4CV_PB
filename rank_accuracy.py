from pyimagesearch.utils.ranked import ranked5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help='path to trained model')
ap.add_argument('-d', '--db', required=True,
                help='path to HDF5 dataset')
args = vars(ap.parse_args())

print('[INFO] loading model...')
model = pickle.loads(open(args['model'], 'rb').read())

print('[INFO] loading datasets...')
db = h5py.File(args['db'], 'r')
i = int(db['labels'].shape[0] * 0.75)

print('[INFO] predicting...')
preds = model.predict_proba(db['features'][i:])
(rank1, rank5) = ranked5_accuracy(preds, db['labels'][i:])

print('[INFO] rank-1 accuracy: {:.2f}%'.format(rank1 * 100))

print('[INFO] rank-5 accuracy: {:.2f}%'.format(rank5 * 100))