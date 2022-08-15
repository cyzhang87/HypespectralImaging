import pathlib,random
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import torch
import torch.utils.data as TorchData
from scipy import ndimage
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold
import pickle
import config


def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

#get reflectance
def get_reflectance(data, data_gt):
    """
    I_white = np.max(data[data_gt == 9], axis=0)
    I_dark = np.min(data[data_gt == 8], axis=0)
    I_w, I_d = np.zeros_like(data), np.zeros_like(data)
    for i in range(I_white.shape[0]):
       I_w[:, :, i] = I_white[i]
       I_d[:, :, i] = I_dark[i]
    """
    I_white = np.max(data[data_gt == 9])
    I_dark = min(np.min(data[data_gt == 7]), np.min(data[data_gt == 8]))
    return (data - I_dark) / (I_white - I_dark)


class Data():
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_file = args.data_file
        self.data_name = args.data_name
        self.train_num = args.train_num
        self.class_num = args.class_num
        self.seed = args.seed
        self.result = args.result
        self.tfrecords = args.tfrecords
        self.args = args
        self.cube_size = args.cube_size
        
        self.data_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_file, self.data_name + '_pre3.mat')))
        self.data_gt_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_file, self.data_name+'_gt.mat')))
        data_name = [t for t in list(self.data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(self.data_gt_dict.keys()) if not t.startswith('__')][0]
        self.data = self.data_dict[data_name].astype(np.int64)
        self.data_gt = self.data_gt_dict[data_gt_name].astype(np.int64)
        self.dim = self.data.shape[2]
        self.cropped_dim = int(self.dim) #36
        print('DataSet %s shape is %s'%(self.data_name,self.data.shape))

    def neighbor_add(self,row, col, w_size=3):  # 给出 row，col和标签，返回w_size大小的cube
        t = w_size // 2
        cube = np.zeros(shape=[w_size, w_size, self.data.shape[2]])
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                if i + row < 0 or i + row >= self.data.shape[0] or j + col < 0 or j + col >= self.data.shape[1]:
                    cube[i + t, j + t] = self.data[row, col]
                else:
                    cube[i + t, j + t] = self.data[i + row, j + col]
        return cube


    def spectral_calibrate(self, data):
        for i in range(2, data.shape[0]-2):
            data[i] = (data[i-2]/4 + data[i-1]/2 + data[i] + data[i+1]/2 + data[i+2]/4) / 2.5
        return data

    def read_data(self):
        data = self.data
        data_gt = self.data_gt

        sio.savemat(os.path.join(self.result,'info.mat'),{
            'shape':self.data.shape,
            #'data':self.data,
            'data_gt':self.data_gt,
            'dim':self.data.shape[2],
            'class_num':self.class_num
        })

        PREPROCESS_SWITCH, PREPROCESS_SWITCH_1, PREPROCESS_SWITCH_2, PREPROCESS_SWITCH_3 = False, False, False, False
        PREPROCESS_SWITCH_4 = False
        if PREPROCESS_SWITCH:
            (w, h) = data_gt.shape
            # 1. dead pixels fixing
            if PREPROCESS_SWITCH_1:
                max_diff = 0
                for d in range(data.shape[2]):
                    for i in range(data_gt.shape[0]):
                        for j in range(data_gt.shape[1]):
                            data_slice = data[:, :, d]
                            
                            if data_gt[i, j] == 0:
                                continue
                            
                            if j == 0 or j == (h - 1):
                                continue
                            diff = data_slice[i, j] - data_slice[i, j-1]
                            if diff > max_diff:
                                max_diff = diff
                            if (diff > 1000) and (abs(data_slice[i, j+1] - data_slice[i, j-1]) < 1000):
                                print("({}, {}, {}): {} - {}".format(i, j, d, data_slice[i, j], data_slice[i, j-1]))
                                data_slice[i, j] = (data_slice[i, j-1] + data_slice[i, j+1]) / 2

                print("max_diff: {}".format(max_diff))
                sio.savemat(str(pathlib.Path(self.data_path, self.data_file, self.data_name + '_pre1.mat')),
                            {'newdata': self.data})
            # 2. median filter: smoothing the spatial pixels
            if PREPROCESS_SWITCH_2:
                for d in range(data.shape[2]):
                    data[:, :, d] = ndimage.median_filter(data[:, :, d], size=(20, 20))
                sio.savemat(str(pathlib.Path(self.data_path, self.data_file, self.data_name + '_pre2.mat')),
                            {'newdata': self.data})
            # 3. SG filter: smoothing the spectral pixels
            if PREPROCESS_SWITCH_3:
                for i in range(data_gt.shape[0]):
                    for j in range(data_gt.shape[1]):
                        if data_gt[i, j] == 0:
                            continue
                        #data[i, j] = savgol_filter(data[i, j], 9, 2)
                        data[i, j] = self.spectral_calibrate(data[i, j])

                sio.savemat(str(pathlib.Path(self.data_path, self.data_file, self.data_name + '_pre3.mat')),
                            {'newdata': self.data})
            if PREPROCESS_SWITCH_4:
                for d in range(data.shape[2]):
                    for c in range(1, self.class_num+1):
                        data_tmp = data[:, :, d][data_gt==c]
                        std = np.std(data_tmp)
                        data[:, :, d][data_gt==c] = np.mean(data_tmp)
                        data[:, :, d][data_gt==c] += np.random.normal(0, std, len(data_tmp)).astype(np.int64)
                sio.savemat(str(pathlib.Path(self.data_path, self.data_file, self.data_name + '_pre4.mat')),
                            {'newdata': self.data})
        #data = max_min(data).astype(np.float32)
        data = get_reflectance(data, data_gt)
        self.data = data
        if config.BINARY_SWITCH == False:
            class_num = self.class_num
        else:
            class_num = len(config.label_range)
        data_pos = {i: [] for i in range(1, class_num + 1)}
        train_pos = {i: [] for i in range(1, class_num + 1)}
        val_pos = {i: [] for i in range(1, class_num + 1)}
        test_pos = {i: [] for i in range(1, class_num + 1)}
        center_pos = {i: [] for i in range(1, class_num + 1)}

        label_range = config.label_range
        if int(self.data_file[-2:]) > 53:
            label_range = (1, 3, 4, 5, 6)

        for i in range(data_gt.shape[0]):
            for j in range(data_gt.shape[1]):
                kk = 1
                #for k in range(1, class_num + 1):
                for k in label_range:
                    if data_gt[i, j] == k:
                        if self.data_name == 'dftc':
                            train_pos[kk].append([i, j])
                        else:
                            data_pos[kk].append([i, j])
                    if self.data_name == 'dftc':
                        if self.test_gt[i,j]==k:
                            test_pos[kk].append([i, j])
                    kk += 1
        self.data_pos = data_pos
        if self.args.fix_seed:
            random.seed(self.seed)
        
        for k, v in data_pos.items():
            if self.train_num > 0 and self.train_num < 1:
                train_num = int(self.train_num * len(v))
            elif len(v)<self.train_num:
                train_num = 15
            else:
                train_num = self.train_num
            train_pos[k] = random.sample(v, int(train_num))
            test_pos[k] = [i for i in v if i not in train_pos[k]]
        self.train_pos = train_pos
        self.test_pos = test_pos
        train_pos_all = list()
        test_pos_all = list()
        for k,v in self.train_pos.items():
            for t in v:
                train_pos_all.append([k,t])
        for k,v in self.test_pos.items():
            for t in v:
                test_pos_all.append([k,t])
        train_t = 0
        test_t = 0
        for (k1,v1),(k2,v2) in zip(self.train_pos.items(),self.test_pos.items()):
            print('traindata-ID %s: %s; testdata-ID %s: %s'%(k1,len(v1),k2,len(v2)))
            train_t += len(v1)
            test_t += len(v2)
        print('total train %s, total test %s'%(train_t,test_t))
        # for k,v in self.test_pos.items():
        #     print('testdata-ID %s: %s'%(k,len(v)))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # train data
        train_data_name = os.path.join(self.tfrecords, 'train_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(train_data_name)
        for i in train_pos_all:
            [r,c] = i[1]
            pixel_t = self.neighbor_add(r,c,w_size=self.cube_size).astype(np.float32).tostring()
            label_t = np.array(np.array(i[0] - 1).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'traindata': _bytes_feature(pixel_t),
                    'trainlabel': _int64_feature(label_t)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()

        # test data
        test_data_name = os.path.join(self.tfrecords, 'test_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(test_data_name)
        for i in test_pos_all:
            [r, c] = i[1]
            pixel_t = self.neighbor_add(r,c,w_size=self.cube_size).astype(np.float32).tostring()
            label_t = np.array(np.array(i[0] - 1).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'testdata': _bytes_feature(pixel_t),
                    'testlabel': _int64_feature(label_t)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()

        """
        # map data
        map_data_name = os.path.join(self.tfrecords, 'map_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(map_data_name)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data_gt[i,j] == 0:
                    continue
                pixel_t = self.neighbor_add(i, j, w_size=self.cube_size).astype(np.float32).tostring()
                pos = [i,j]
                pos = np.asarray(pos,dtype=np.int64).tostring()
                example = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'mapdata': _bytes_feature(pixel_t),
                        'pos': _bytes_feature(pos),
                    }
                )))
                writer.write(example.SerializeToString())
        writer.close()

        # map seg data
        map_data_name = os.path.join(self.tfrecords, 'map_data_seg.tfrecords')
        writer = tf.python_io.TFRecordWriter(map_data_name)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # if data_gt[i,j] == 0:
                #     continue
                pixel_t = self.neighbor_add(i, j, w_size=self.cube_size).astype(np.float32).tostring()
                pos = [i,j]
                pos = np.asarray(pos,dtype=np.int64).tostring()
                example = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'mapdata': _bytes_feature(pixel_t),
                        'pos': _bytes_feature(pos),
                    }
                )))
                writer.write(example.SerializeToString())
        writer.close()
        """


    def data_parse(self, filename, type='train'):
        dataset = tf.data.TFRecordDataset([filename])
        # create training data loader
        def parser_train(record):
            keys_to_features = {
                'traindata': tf.FixedLenFeature([], tf.string),
                'trainlabel': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            train_data = tf.decode_raw(features['traindata'], tf.float32)
            train_label = tf.cast(features['trainlabel'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            train_data = tf.reshape(train_data, shape)
            train_label = tf.reshape(train_label, [1])
            return train_data, train_label
        def parser_test(record):
            keys_to_features = {
                'testdata': tf.FixedLenFeature([], tf.string),
                'testlabel': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            test_data = tf.decode_raw(features['testdata'], tf.float32)
            test_label = tf.cast(features['testlabel'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            test_data = tf.reshape(test_data, shape)
            test_label = tf.reshape(test_label, [1])
            return test_data, test_label
        def parser_map(record):
            keys_to_features = {
                'mapdata': tf.FixedLenFeature([], tf.string),
                'pos': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            map_data = tf.decode_raw(features['mapdata'], tf.float32)
            pos = tf.decode_raw(features['pos'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            map_data = tf.reshape(map_data, shape)
            pos = tf.reshape(pos,[2])
            return map_data,pos
        def parser_map_seg(record):
            keys_to_features = {
                'mapdata': tf.FixedLenFeature([], tf.string),
                'pos': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            map_data = tf.decode_raw(features['mapdata'], tf.float32)
            pos = tf.decode_raw(features['pos'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            map_data = tf.reshape(map_data, shape)
            pos = tf.reshape(pos,[2])
            return map_data,pos

        if type == 'train':
            dataset = dataset.map(parser_train)
            dataset = dataset.shuffle(buffer_size=20000)
            dataset = dataset.batch(self.args.batch_size)
            #dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            element = iterator.get_next()
            train_data = np.array([])
            train_label = np.array([])
            with tf.Session() as sess:
                while True:
                    try:
                        train_data_tmp,train_label_tmp = sess.run(element)
                        if train_data.shape[0] == 0:
                            train_data = train_data_tmp
                            train_label = train_label_tmp
                        else:
                            train_data = np.concatenate((train_data, train_data_tmp))
                            train_label = np.append(train_label, train_label_tmp)
                    except tf.errors.OutOfRangeError:
                        print("数据读取完毕")
                        break

            # data aug
            if config.DATA_AUG_SWITCH:
                labels = [3, 4]
                train_data_aug = train_data.copy()
                train_label_aug = train_label.copy()
                Y = np.mean(np.mean(np.mean(train_data_aug[train_label_aug == 0], axis=0), axis=0), axis=0)
                data1_range = np.array([[0.000940, 0.228590], [0.001034, 0.208818], [-0.000606, 0.203702], [-0.000798, 0.192845], [-0.000593, 0.184436]])
                data2_range = np.array([[0.000777, 0.229236], [0.000807, 0.209079], [-0.000576, 0.213001], [-0.000936, 0.192830], [-0.001098, 0.185824]])
                for d in range(train_data_aug.shape[3]):
                    for c in range(data1_range.shape[0]):
                        data_tmp = train_data_aug[:, :, :, d][train_label_aug == c]
                        train_data_aug[:, :, :, d][train_label_aug == c] = data2_range[c, 0] + (data_tmp - data1_range[c, 0]) * (data2_range[c, 1] - data2_range[c, 0]) / (data1_range[c, 1] - data1_range[c, 0])

                train_data_aug_2 = train_data.copy()
                data2_range = np.array([[0.001330, 0.235119], [0.001350, 0.218434], [-0.000164, 0.212817], [-0.000312, 0.198926], [-0.000380, 0.194064]])
                for d in range(train_data_aug_2.shape[3]):
                    for c in range(data1_range.shape[0]):
                        data_tmp = train_data_aug_2[:, :, :, d][train_label_aug == c]
                        train_data_aug_2[:, :, :, d][train_label_aug == c] = data2_range[c, 0] + (data_tmp - data1_range[c, 0]) * (data2_range[c, 1] - data2_range[c, 0]) / (data1_range[c, 1] - data1_range[c, 0])

                train_data = np.concatenate((train_data, train_data_aug, train_data_aug_2))
                train_label = np.concatenate((train_label, train_label_aug, train_label_aug))
            # data aug end

            # binary the dataset
            if config.BINARY_SWITCH:
                for label in (1, 2, 3, 4, 5):
                    train_label[train_label == label] = 1
            # binary end
            random_state = np.random.RandomState(0)
            train_data = train_data[:, :, :, :]

            if config.KFOLD_SWITCH:
                random_state = np.random.RandomState(0)
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
                k_index = 0
                for train_index, test_index in skf.split(train_data, train_label):
                    pickle.dump({'data': train_data[train_index], 'label': train_label[train_index]},
                                open(os.path.join(self.args.tfrecords, 'kfold_train_' + str(k_index)) + '.pkl', 'wb'))
                    pickle.dump({'data': train_data[test_index], 'label': train_label[test_index]},
                                open(os.path.join(self.args.tfrecords, 'kfold_val_' + str(k_index)) + '.pkl', 'wb'))
                    k_index += 1

            X_train, X_val, y_train, y_val = train_test_split(train_data,
                                                                train_label,
                                                                test_size=0.2,
                                                                random_state=random_state)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            torch_dataset = TorchData.TensorDataset(X_train, y_train)
            train_loader = TorchData.DataLoader(
                dataset=torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
            )

            val_data_dict = {}
            val_data_dict['data'] = torch.tensor(X_val, dtype=torch.float32)
            val_data_dict['label'] = torch.tensor(y_val, dtype=torch.long)

            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
            torch_dataset = TorchData.TensorDataset(X_val, y_val)
            val_loader = TorchData.DataLoader(
                dataset=torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            print('train data size: ' + str(X_train.shape))
            return train_loader, val_loader, val_data_dict

        if type == 'test':
            dataset = dataset.map(parser_test)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            element = iterator.get_next()
            test_data = np.array([])
            test_label = np.array([])
            with tf.Session() as sess:
                while True:
                    try:
                        test_data_tmp, test_label_tmp = sess.run(element)
                        if test_data.shape[0] == 0:
                            test_data = test_data_tmp
                            test_label = test_label_tmp
                        else:
                            test_data = np.concatenate((test_data, test_data_tmp))
                            test_label = np.append(test_label, test_label_tmp)
                    except tf.errors.OutOfRangeError:
                        print("数据读取完毕")
                        break
            if len(test_label.shape) == 2:
                test_label = test_label[:,0]

            # binary the dataset
            if config.BINARY_SWITCH:
                for label in (1, 2, 3, 4, 5):
                    test_label[test_label == label] = 1
            # binary end

            test_data = test_data[:, :, :, :]
            test_data_dict = {}
            test_data_dict['data'] = torch.tensor(test_data, dtype=torch.float32)
            test_data_dict['label'] = torch.tensor(test_label, dtype=torch.long)
            test_data = torch.tensor(test_data, dtype=torch.float32)
            test_label = torch.tensor(test_label, dtype=torch.long)
            torch_dataset = TorchData.TensorDataset(test_data, test_label)
            test_loader = TorchData.DataLoader(
                dataset=torch_dataset,
                batch_size=self.args.test_batch,
                shuffle=False,
                num_workers=0,
            )
            print('test data size: ' + str(test_data.shape))
            return test_loader, test_data_dict

        if type == 'kfold':
            data_dict = pickle.load(open(filename, 'rb'))
            X = torch.tensor(data_dict['data'], dtype=torch.float32)
            y = torch.tensor(data_dict['label'], dtype=torch.long)
            torch_dataset = TorchData.TensorDataset(X, y)
            data_loader = TorchData.DataLoader(
                dataset=torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
            )

            data_dict['data'] = X
            data_dict['label'] = y

            print('data size: ' + str(X.shape))
            return data_loader, data_dict

        if type == 'map':
            dataset = dataset.map(parser_map).repeat(1)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'map_seg':
            dataset = dataset.map(parser_map_seg).repeat(1)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
