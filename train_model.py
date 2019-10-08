from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', required=True, help='path to dataset')
ap.add_argument('-o', '--output', required=True, help='path to save output model')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='number of jobs to run when '
                                                           'tuning hyper parameter')
args = vars(ap.parse_args())

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(args['db'], 'r')
i = int(db['labels'].shape[0] * 0.75)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyper-parameters...")
pararms = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
model = GridSearchCV(LogisticRegression(), param_grid=pararms,
                     cv=3, n_jobs=args['jobs'])
model.fit(db['features'][:i], db['labels'][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

print("[INFO] evaluating...")
preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], preds, target_names=db['label_names']))

print('[INFO] serializing model...')
f = open(args['output'], 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()