import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey='images', buffSize=1000):

        # check if output path exist raise exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already exist and can not be overwritten. "
                             "Manually delete file before continuing.")

        # open hdf5 database and create two datasets
        # one for class labels
        # and other for images/features
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(name=dataKey, shape=dims,
                                           dtype='float')
        self.label = self.db.create_dataset(name='labels', shape=(dims[0],),
                                            dtype='int')

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.buffSize = buffSize
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to buffer size
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        if len(self.buffer['data']) >= self.buffSize:
            self.flush()

    def flush(self):
        # write the buffer to list then reset buffer
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.label[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def storeClassLabels(self, classLabels):
        dt = h5py.string_dtype()
        labelSet = self.db.create_dataset(name="label_names", shape=(len(classLabels),),
                                          dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer['data']) > 0:
            self.flush()

        self.db.close()
