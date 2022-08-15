# encoding: utf-8
import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.models import Conv2dModel, Conv3dModel, BidirectionalLSTM, BidirectionalGRU, Conv3d_RNN, JointDeepModel, MLayerBiLSTMModel, FMC
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import SqueezeNet
from models.resnet import resnet34 as ResNet34_3D
from models.resnext import resnext50
from models.resnet2 import resnet34 as ResNet34_2D
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import scipy.io as sio
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import pickle
import copy
import config
from utils.data_loader_v2 import Data

config.set_mode('five_grade_mode')
pd.set_option('display.max_columns', 10)

parser = argparse.ArgumentParser(description='2D CNN for hyperspectral image classification')
parser.add_argument('-me', '--method', help='method, 2D_CNN, 3D_CNN, ShuffleNetV2, Conv3d_LSTM, Conv3d_GRU, JointDeepModel, MixModel, SqueezeNet, Resnet, Resnext, ResNet34_2D'
                                            'SVM, RandomForest, GradientBoosting, LogisticRegression, SVM-RBF, LDA', default='RNN')
parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--log',dest='log',default='log')
parser.add_argument('--model',dest='model',default='model')
parser.add_argument('--tfrecords',dest='tfrecords',default='tfrecords')
parser.add_argument('--data_path',dest='data_path',default="./data/")
parser.add_argument('--data_file',dest='data_file',default="SWIR_100051") #jiaoda1112_1_json, SWIR_100051, VNIR_100051, SWIR_100051_2记录了数据平滑过程
parser.add_argument('--data_name',dest='data_name',default='data')
parser.add_argument('--use_lr_decay',dest='use_lr_decay',default=True)
parser.add_argument('--decay_rate',dest='decay_rate',default=0.90)
parser.add_argument('--decay_steps',dest='decay_steps',default=10000)
parser.add_argument('--learning_rate',dest='lr',default=0.001)
parser.add_argument('--train_num',dest='train_num',default=0.7) # 250 intger for number and decimal for percentage
parser.add_argument('--batch_size',dest='batch_size',default=128)
parser.add_argument('--fix_seed',dest='fix_seed',default=False)
parser.add_argument('--seed',dest='seed',default=666)
parser.add_argument('--class_num',dest='class_num',default=config.class_num)
parser.add_argument('--test_batch',dest='test_batch',default=128)
parser.add_argument('--iter_num',dest='iter_num',default=30000)
parser.add_argument('--cube_size',dest='cube_size',default=3)
parser.add_argument('--save_decode_map',dest='save_decode_map',default=True)
parser.add_argument('--save_decode_seg_map',dest='save_decode_seg_map',default=True)
parser.add_argument('--load_model',dest='load_model',default=False)
parser.add_argument('--eval', type=str, default='train')
args = parser.parse_args()

if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.tfrecords):
    os.mkdir(args.tfrecords)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

FMC_Str = 'FMC_LDA_RF'

#os.environ['CUDA_VISIBLE_DEVICES']= '0'


class RNNMethod:
    def __init__(self, method, args):
        self.data_name = args.data_name
        self.result = args.result
        info = sio.loadmat(os.path.join(self.result, 'info.mat'))
        self.shape = info['shape']
        self.dim = int(info['dim'])
        self.cropped_dim = int(self.dim)
        self.class_num = args.class_num
        self.data_gt = info['data_gt']
        self.log = args.log
        self.model = args.model
        self.cube_size = args.cube_size
        self.data_path = args.data_path
        self.iter_num = args.iter_num
        self.tfrecords = args.tfrecords
        self.best_oa = 0

        if method == '2D_CNN':
            self.model = Conv2dModel(self.cropped_dim, (3, 3), self.class_num).to(device)
        elif method == '3D_CNN':
            self.model = Conv3dModel(1, self.cropped_dim, (3, 3, 3), self.class_num).to(device)
        elif method == 'Conv3d_LSTM':
            self.model = Conv3d_RNN(1, (3, 3, 3), self.class_num, 'lstm').to(device)
        elif method == 'Conv3d_GRU':
            self.model = Conv3d_RNN(1, (3, 3, 3), self.class_num, 'gru').to(device)
        elif method == 'RNN':
            self.model = MLayerBiLSTMModel(1, self.class_num).to(device)
        elif method == 'ShuffleNetV2':
            self.model = ShuffleNetV2(nIn=1, nOut=self.class_num).to(device)
        elif method == 'JointDeepModel':
            self.model = JointDeepModel(nIn_2d=self.cropped_dim, nIn_3d=1,
                                        nkernel_2d=(3, 3), nkernel_3d=(3, 3, 3),
                                        nOut=self.class_num, rnn_type='gru').to(device)
        elif method == FMC_Str:
            self.model = FMC(nIn_2d=self.cropped_dim, nOut=self.class_num).to(device)
        elif method == 'SqueezeNet':
            self.model = SqueezeNet(ndim=self.cropped_dim, num_classes=self.class_num).to(device)
        elif method == 'ResNet34_3D':
            self.model = ResNet34_3D(ndim=self.cropped_dim, num_classes=self.class_num).to(device)
        elif method == 'Resnext':
            self.model = resnext50(ndim=self.cropped_dim, num_classes=self.class_num).to(device)
        elif method == 'ResNet34_2D':
            self.model = ResNet34_2D(n_classes=self.class_num, input_channels=272, avg_ks=args.cube_size).to(device)
        print(self.model)
        self.loss_fn = nn.CrossEntropyLoss()

        #self.loss_fn = F.nll_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.decay_rate)

    def train(self, loader, epoch, premodels=None):
        self.model.train()
        for step, (batch_x, batch_y) in enumerate(loader):
            if len(batch_y.shape) == 2:
                batch_y = batch_y[:, 0]

            if args.method == FMC_Str:
                features = np.array([])
                for index in range(len(premodels)):
                    feature = self.get_batch_features(premodels[index], batch_x.to(device))
                    if features.shape[0] == 0:
                        features = feature
                    else:
                        features = np.concatenate((features, feature), axis=1)
                batch_x = torch.tensor(features)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred, _ = self.model(batch_x)
            loss = self.loss_fn(pred, batch_y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.my_lr_scheduler.step()

        if epoch % 100 == 0:
            print('epoch: {}'.format(epoch))
            acc = 100 * (pred.argmax(1) == batch_y).type(torch.float).sum().item() / batch_y.shape[0]
            macro_f1 = f1_score(batch_y.cpu(), pred.argmax(1).cpu(), average='macro')
            print(f"Train: accuracy: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

        #return loss, acc, macro_f1
        return self.model


    def softmax(self, x):
        max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
        e_x = np.exp(x - max)  # subtracts each row with its max value
        sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x

    def val(self, loader, premodels=None):
        self.model.eval()
        labels = np.array([])
        preds = []
        for i in range(args.class_num):
            preds.append([])
        preds = np.array(preds).T
        for step, (batch_x, batch_y) in enumerate(loader):
            if len(batch_y.shape) == 2:
                batch_y = batch_y[:, 0]

            if args.method == FMC_Str:
                features = np.array([])
                for index in range(len(premodels)):
                    feature = self.get_batch_features(premodels[index], batch_x.to(device))
                    if features.shape[0] == 0:
                        features = feature
                    else:
                        features = np.concatenate((features, feature), axis=1)
                batch_x = torch.tensor(features)

            # Compute prediction error
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred, _ = self.model(batch_x)
            labels = np.concatenate((labels, batch_y.cpu().numpy()))
            preds = np.concatenate((preds, pred.to('cpu').detach().numpy()))

        loss = self.loss_fn(torch.tensor(preds, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)).item()
        #loss_onehot = self.loss_fn(pred_onehot, dataset['label']).item()
        acc = 100 * (preds.argmax(1) == labels).sum().item() / labels.shape[0]
        macro_f1 = f1_score(labels, preds.argmax(1), average='macro')
        #print(f"Validate acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")
        return loss, acc, macro_f1

    def metrics(self, preds, labels):
        pred_tmp = preds.copy()
        preds = preds.argmax(1)
        loss = self.loss_fn(torch.tensor(pred_tmp, dtype=torch.float32),
                                     torch.tensor(labels, dtype=torch.long)).item()
        acc = 100 * (preds == labels).sum().item() / labels.shape[0]
        macro_f1 = f1_score(labels, preds, average='macro')
        print(f"Test accuracy: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")
        matrix = confusion_matrix(labels, preds)

        ac_list = []
        for i in range(len(matrix)):
            ac = round(matrix[i, i] / sum(matrix[i, :]), 4)
            ac_list.append(ac)
            print(i + 1, 'class:', '(', matrix[i, i], '/', sum(matrix[i, :]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print(f'oa: {accuracy:0.4f}')
        # kappa
        kappa_value = cohen_kappa_score(labels, preds)
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[i, :])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print(f'aa: {aa:0.4f}')
        print(f'kappa: {kappa:0.4f}')

        matrix_pro = np.round(matrix / np.sum(matrix, axis=1), 4)
        return oa, aa, kappa, ac_list, loss, macro_f1, pred_tmp, labels, matrix_pro

    def test(self, loader, premodels=None):
        self.model.eval()
        with torch.no_grad():
            labels = np.array([])
            preds = []
            for i in range(args.class_num):
                preds.append([])
            preds = np.array(preds).T
            for step, (batch_x, batch_y) in enumerate(loader):
                if len(batch_y.shape) == 2:
                    batch_y = batch_y[:, 0]

                if args.method == FMC_Str:
                    features = np.array([])
                    for index in range(len(premodels)):
                        feature = self.get_batch_features(premodels[index], batch_x.to(device))
                        if features.shape[0] == 0:
                            features = feature
                        else:
                            features = np.concatenate((features, feature), axis=1)
                    batch_x = torch.tensor(features)

                # Compute prediction error
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred, _ = self.model(batch_x)
                labels = np.concatenate((labels, batch_y.cpu().numpy()))
                preds = np.concatenate((preds, pred.to('cpu').detach().numpy()))

        return self.metrics(preds, labels)
        #return oa, aa, kappa, ac_list, loss, macro_f1, preds

    def voting_test(self, loader, models_weights, voting_models=None):
        self.model.eval()
        with torch.no_grad():
            labels = np.array([])
            models_preds = np.array([])
            for step, (batch_x, batch_y) in enumerate(loader):
                if len(batch_y.shape) == 2:
                    batch_y = batch_y[:, 0]

                models_pred = []
                for index in range(len(voting_models)):
                    pred, _ = voting_models[index].model(batch_x.to(device))
                    models_pred.append(self.softmax(pred.cpu().detach().numpy()) * models_weights[index])
                models_pred = np.array(models_pred)
                models_pred = np.sum(models_pred, axis=0)
                if models_preds.shape[0] == 0:
                    models_preds = models_pred
                else:
                    models_preds = np.concatenate((models_preds, models_pred))
                labels = np.concatenate((labels, batch_y.cpu().numpy()))

        return self.metrics(models_preds, labels)


    def get_features(self, dataset):
        self.model.eval()
        _, features = self.model(dataset['data'].to(device))
        return features.detach().cpu().numpy()


    def get_batch_features(self, model, data):
        model.model.eval()
        _, features = model.model(data)
        return features.detach().cpu().numpy()


def train_tradition_algs(algorithm, args, loader, best_model, premodels=None):
    X_origin = np.array([])
    X = np.array([])
    y = np.array([])
    if algorithm == 'MixModel':
        method = RNNMethod('JointDeepModel', args)
        dlmodel = best_model[:best_model.find('model_') + len('model_')] + 'JointDeepModel.pth'
        method.model.load_state_dict(torch.load(dlmodel, map_location='cpu'))
        method.model.eval()

    for step, (batch_x, batch_y) in enumerate(loader):
        if algorithm == 'MixModel':
            _, feature = method.model(torch.tensor(batch_x).to(device))
            feature = feature.to('cpu').detach().numpy()
            if X.shape[0] == 0:
                X = feature
                y = batch_y
            else:
                X = np.concatenate((X, feature))
                y = np.append(y, batch_y)
        elif algorithm == FMC_Str:
            features = np.array([])
            for index in range(len(premodels)):
                feature = premodels[index].get_batch_features(premodels[index], batch_x.to(device))
                if features.shape[0] == 0:
                    features = feature
                else:
                    features = np.concatenate((features, feature), axis=1)
            if X.shape[0] == 0:
                X = features
                y = batch_y
            else:
                X = np.concatenate((X, features))
                y = np.append(y, batch_y)
        else:
            batch_x = batch_x.numpy()
            batch_y = batch_y.numpy()
            if X_origin.shape[0] == 0:
                X_origin = batch_x
                y = batch_y
            else:
                X_origin = np.concatenate((X_origin, batch_x))
                y = np.append(y, batch_y)
            X = np.mean(np.mean(X_origin, axis=1), axis=1)
    if algorithm == 'SVM-RBF':
        model = svm.SVC(kernel='rbf')
    elif algorithm == 'LogisticRegression':
        model = LogisticRegression(random_state=0, multi_class='auto')
    elif algorithm == 'RandomForest':
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=200))
        #model = RandomForestClassifier(n_estimators=200)
    elif algorithm == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=200)
    elif algorithm == 'LDA':
        lda = LDA(n_components=4)
        X = lda.fit_transform(X, y)
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)
        pickle.dump([model, lda], open(best_model, 'wb'))
        return
    elif algorithm == 'MixModel':
        model = RandomForestClassifier(n_estimators=200)
    elif algorithm == FMC_Str:
        lda = LDA(n_components=4)
        X = lda.fit_transform(X, y)
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)
        pickle.dump([model, lda], open(best_model, 'wb'))
        return

    model.fit(X, y)
    pickle.dump(model, open(best_model, 'wb'))


def test_tradition_algs(algorithm, args, test_data, best_model, loader, premodels=None):
    X_test = torch.mean(torch.mean(test_data['data'], axis=1), axis=1).numpy()
    y_test = test_data['label'].numpy()
    if algorithm == 'LDA':
        [loaded_model, lda] = pickle.load(open(best_model, 'rb'))
        X_test = lda.transform(X_test)
    elif algorithm == 'MixModel':
        method = RNNMethod('JointDeepModel', args)
        dlmodel = best_model[:best_model.find('model_') + len('model_')] + 'JointDeepModel.pth'
        method.model.load_state_dict(torch.load(dlmodel, map_location='cpu'))
        method.model.eval()
        with torch.no_grad():
            y_test = np.array([])
            X_test = np.array([])
            for step, (batch_x, batch_y) in enumerate(loader):
                if len(batch_y.shape) == 2:
                    batch_y = batch_y[:, 0]
                _, feature = method.model(batch_x)
                y_test = np.concatenate((y_test, batch_y.cpu().numpy()))
                if X_test.shape[0] == 0:
                    X_test = feature.to('cpu').detach().numpy()
                else:
                    X_test = np.concatenate((X_test, feature.to('cpu').detach().numpy()))
    elif algorithm == FMC_Str:
        with torch.no_grad():
            y_test = np.array([])
            X_test = np.array([])
            for step, (batch_x, batch_y) in enumerate(loader):
                if len(batch_y.shape) == 2:
                    batch_y = batch_y[:, 0]
                batch_x = batch_x.to(device)
                features = np.array([])
                for index in range(len(premodels)):
                    feature = premodels[index].get_batch_features(premodels[index], batch_x.to(device))
                    if features.shape[0] == 0:
                        features = feature
                    else:
                        features = np.concatenate((features, feature), axis=1)
                y_test = np.concatenate((y_test, batch_y.cpu().numpy()))
                if X_test.shape[0] == 0:
                    X_test = features
                else:
                    X_test = np.concatenate((X_test, features))
        [loaded_model, lda] = pickle.load(open(best_model, 'rb'))
        X_test = lda.transform(X_test)
    else:
        loaded_model = pickle.load(open(best_model, 'rb'))

    preds = loaded_model.predict(X_test)
    ac = loaded_model.score(X_test, y_test)

    acc = 100 * (preds == y_test).sum().item() / y_test.shape[0]
    macro_f1 = f1_score(y_test, preds, average='macro')
    print(f"Test accuracy: {acc :>0.2f}%, macro-f1: {macro_f1:>8f}")
    matrix = confusion_matrix(y_test, preds)

    ac_list = []
    for i in range(len(matrix)):
        ac = round(matrix[i, i] / sum(matrix[i, :]), 4)
        ac_list.append(ac)
        print(i + 1, 'class:', '(', matrix[i, i], '/', sum(matrix[i, :]), ')', ac)
    print('confusion matrix:')
    print(np.int_(matrix))
    print('total right num:', np.sum(np.trace(matrix)))
    accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
    print(f'oa: {accuracy:0.4f}')
    # kappa
    kappa_value = cohen_kappa_score(y_test, preds)
    kk = 0
    for i in range(matrix.shape[0]):
        kk += np.sum(matrix[i]) * np.sum(matrix[i, :])
    pe = kk / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    ac_list = np.asarray(ac_list)
    aa = np.mean(ac_list)
    oa = accuracy
    print(f'aa: {aa:0.4f}')
    print(f'kappa: {kappa:0.4f}')
    matrix_pro = np.round(matrix / np.sum(matrix, axis=1), 4)
    return oa, aa, kappa, ac_list, 0, macro_f1, preds, matrix_pro, matrix, y_test

def is_tradition_alg(algorithm):
    if (algorithm == 'SVM-RBF') or \
            (algorithm == 'LogisticRegression') or \
            (algorithm == 'RandomForest') or \
            (algorithm == 'GradientBoosting') or \
            (algorithm == 'LDA') or \
            (algorithm == FMC_Str) or \
            (algorithm == 'MixModel'):
        return True

def single_process(algorithm):
    test_score_list = []
    test_pred_list = []
    args.model = os.path.join(args.model, args.data_file)
    args.log = os.path.join(args.log, args.data_file)
    args.result = os.path.join(args.result, args.data_file)
    args.tfrecords = os.path.join(args.tfrecords, args.data_file)
    if not os.path.exists(args.model):
        os.mkdir(args.model)
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    if not os.path.exists(args.tfrecords):
        os.mkdir(args.tfrecords)

    total_test_time = datetime.now() - datetime.now()
    loop_round = 8
    for loop in range(7, loop_round):
        print('{}: loop {}'.format(datetime.now(), loop))
        args.id = str(loop)
        args.result = os.path.join(args.result, args.id)
        args.log = os.path.join(args.log, args.id)
        args.model = os.path.join(args.model, args.id)
        args.tfrecords = os.path.join(args.tfrecords, args.id)
        if not os.path.exists(args.model):
            os.mkdir(args.model)
        if not os.path.exists(args.log):
            os.mkdir(args.log)
        if not os.path.exists(args.result):
            os.mkdir(args.result)
        if not os.path.exists(args.tfrecords):
            os.mkdir(args.tfrecords)
        dataset = Data(args)
        if not os.path.exists(os.path.join(args.tfrecords, 'train_data.tfrecords')):
            dataset.read_data()
        best_model = args.model + '/model_{}.pth'.format(args.method)
        test_data_loader, test_data = dataset.data_parse(os.path.join(args.tfrecords, 'test_data.tfrecords'), type='test')
        if not os.path.exists(best_model):
            train_data_loader, val_data_loader, _ = dataset.data_parse(os.path.join(args.tfrecords, 'train_data.tfrecords'), type='train')
            #val_data_loader, _ = dataset.data_parse(os.path.join(args.tfrecords, 'val_data.tfrecords'), type='test')
            if is_tradition_alg(algorithm):
                train_tradition_algs(algorithm, args, train_data_loader, best_model)
            else:
                loss_min, f1_max, acc_max, es_count, epoch_max = 1000, 0, 0, 0, 0
                method = RNNMethod(algorithm, args)
                method.best_oa = 0
                for epoch in range(args.iter_num):
                    #print('epoch: {}'.format(epoch))
                    method.train(train_data_loader, epoch)
                    loss_val, acc_val, f1_val = method.val(val_data_loader)
                    if epoch % 1000 == 0:
                        loss, acc, macro_f1 = method.val(test_data_loader)
                        print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                    if loss_val <= loss_min:
                        loss_min = loss_val
                        acc_max = acc_val
                        f1_max = f1_val
                        epoch_max = epoch
                        # saving model
                        torch.save(method.model.state_dict(), best_model)
                        print("Validate epoch: {} acc: {:>0.2f}%, loss: {:>8f}, macro-f1: {:>8f}, Saved PyTorch Model State to model.pth".format(epoch_max, acc_max, loss_min, f1_max))
                        loss, acc, macro_f1 = method.val(test_data_loader)
                        print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                #torch.save(method.model.state_dict(), best_model)
                print("best performance: epoch {}, loss: {:.6f}, acc: {:.2f}%,  f1: {:.6f}".format(epoch_max, loss_min, acc_max, f1_max))
                #val_score_list.append([epoch_max, loss_min, acc_max, f1_max])

        if is_tradition_alg(algorithm):
            oa, aa, kappa, ac_list, loss, f1, pred, matrix_pro, matrix = test_tradition_algs(algorithm, args, test_data, best_model, test_data_loader)
        else:
            method = RNNMethod(algorithm, args)
            method.model.load_state_dict(torch.load(best_model, map_location='cpu'))
            #start_time = datetime.now()
            oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = method.test(test_data_loader)
            #end_time = datetime.now()
        test_score_list.append([algorithm, args.data_file, loss, oa, aa, kappa, ac_list, f1])

    print('test_score_list:')
    df = pd.DataFrame(data=test_score_list, columns=['algorithm', 'data_file', 'loss', 'oa', 'aa', 'kappa', 'ac_list', 'f1'])
    print(df)

def multi_process(algorithm, data_type='SWIR'):
    test_score_list = []
    matrix_pro_list = []
    total_test_time = datetime.now() - datetime.now()
    loop = config.file_no
    if data_type == 'SWIR':
        data_file_list = ['SWIR_100051', 'SWIR_100052', 'SWIR_100053']
    else:
        data_file_list = ['VNIR_100051', 'VNIR_100052', 'VNIR_100053']
    args_copy = copy.deepcopy(args)

    for data_file in data_file_list:
        args_copy.data_file = data_file
        args_copy.model = os.path.join(args.model, data_file)
        args_copy.log = os.path.join(args.log, data_file)
        args_copy.result = os.path.join(args.result, data_file)
        args_copy.tfrecords = os.path.join(args.tfrecords, data_file)

        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        print('{}: loop {}'.format(datetime.now(), loop))
        args_copy.id = str(loop)
        args_copy.result = os.path.join(args_copy.result, args_copy.id)
        args_copy.log = os.path.join(args_copy.log, args_copy.id)
        args_copy.model = os.path.join(args_copy.model, args_copy.id)
        args_copy.tfrecords = os.path.join(args_copy.tfrecords, args_copy.id)
        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)
        dataset = Data(args_copy)
        if not os.path.exists(os.path.join(args_copy.tfrecords, 'train_data.tfrecords')):
            dataset.read_data()
        #best_model = args_copy.model + '/model_{}.pth'.format(args_copy.method)
        best_model = os.path.join(args.model, data_file_list[0], str(loop)) + '/model_{}.pth'.format(args_copy.method)
        test_data_loader, test_data = dataset.data_parse(os.path.join(args_copy.tfrecords, 'test_data.tfrecords'), type='test')
        if not os.path.exists(best_model):
            train_data_loader, val_data_loader, _ = dataset.data_parse(os.path.join(args_copy.tfrecords, 'train_data.tfrecords'), type='train')
            #val_data_loader, _ = dataset.data_parse(os.path.join(args.tfrecords, 'val_data.tfrecords'), type='test')
            if is_tradition_alg(algorithm):
                train_tradition_algs(algorithm, args_copy, train_data_loader, best_model)
            else:
                loss_min, f1_max, acc_max, es_count, epoch_max = 1000, 0, 0, 0, 0
                method = RNNMethod(algorithm, args_copy)
                method.best_oa = 0
                for epoch in range(args_copy.iter_num):
                    #print('epoch: {}'.format(epoch))
                    method.train(train_data_loader, epoch)
                    loss_val, acc_val, f1_val = method.val(val_data_loader)
                    if epoch % 1000 == 0:
                        loss, acc, macro_f1 = method.val(test_data_loader)
                        print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                    if loss_val <= loss_min:
                        loss_min = loss_val
                        acc_max = acc_val
                        f1_max = f1_val
                        epoch_max = epoch
                        # saving model
                        torch.save(method.model.state_dict(), best_model)
                        print("Validate epoch: {} acc: {:>0.2f}%, loss: {:>8f}, macro-f1: {:>8f}, Saved PyTorch Model State to model.pth".format(epoch_max, acc_max, loss_min, f1_max))
                        loss, acc, macro_f1 = method.val(test_data_loader)
                        print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                #torch.save(method.model.state_dict(), best_model)
                print("best performance: epoch {}, loss: {:.6f}, acc: {:.2f}%,  f1: {:.6f}".format(epoch_max, loss_min, acc_max, f1_max))
                #val_score_list.append([epoch_max, loss_min, acc_max, f1_max])

        print(data_file)
        if is_tradition_alg(algorithm):
            oa, aa, kappa, ac_list, loss, f1, pred, matrix_pro = test_tradition_algs(algorithm, args_copy, test_data, best_model, test_data_loader)
        else:
            method = RNNMethod(algorithm, args_copy)
            method.model.load_state_dict(torch.load(best_model, map_location='cpu'))
            oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = method.test(test_data_loader)
        test_score_list.append([algorithm, data_file, oa, aa, kappa, ac_list, loss, f1])
        matrix_pro_list.append(matrix_pro)

    print('test_score_list:')
    df = pd.DataFrame(data=test_score_list, columns=['algorithm', 'data_file', 'oa', 'aa', 'kappa', 'acc_list', 'loss', 'f1'])
    print(df)
    df.to_csv(os.path.join(args.result, data_type + '_' + str(loop) + '_' + algorithm + '_test_scores.csv'), index=False)
    pickle.dump(matrix_pro_list, open(os.path.join(args.result, data_type + '_' + str(loop) + '_' + algorithm + '_matrix_probabilities.pkl'), 'wb'))


def multi_process_multi_loop(algorithm, data_type='SWIR'):
    test_score_list = []
    matrix_pro_list = []
    total_test_time = datetime.now() - datetime.now()
    file_no = config.file_no
    if data_type == 'SWIR':
        data_file_list = ['SWIR_100051', 'SWIR_100052', 'SWIR_100053']
    else:
        data_file_list = ['VNIR_100051', 'VNIR_100052', 'VNIR_100053']
    args_copy = copy.deepcopy(args)

    for data_file in data_file_list:
        args_copy.data_file = data_file
        args_copy.model = os.path.join(args.model, data_file)
        args_copy.log = os.path.join(args.log, data_file)
        args_copy.result = os.path.join(args.result, data_file)
        args_copy.tfrecords = os.path.join(args.tfrecords, data_file)

        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        print('{}: file_no {}'.format(datetime.now(), file_no))
        args_copy.id = str(file_no)
        args_copy.result = os.path.join(args_copy.result, args_copy.id)
        args_copy.log = os.path.join(args_copy.log, args_copy.id)
        args_copy.model = os.path.join(args_copy.model, args_copy.id)
        args_copy.tfrecords = os.path.join(args_copy.tfrecords, args_copy.id)
        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        for loop in range(10):
            args_copy_loop = copy.deepcopy(args_copy)
            args_copy_loop.tfrecords = os.path.join(args_copy.tfrecords, str(loop))
            if not os.path.exists(args_copy_loop.tfrecords):
                os.mkdir(args_copy_loop.tfrecords)
            dataset = Data(args_copy_loop)

            if not os.path.exists(os.path.join(args_copy_loop.tfrecords, 'train_data.tfrecords')):
                dataset.read_data()
            #best_model = args_copy.model + '/model_{}.pth'.format(args_copy.method)
            best_model = os.path.join(args.model, data_file_list[0], str(file_no)) + '/model_{}_{}.pth'.format(args_copy_loop.method, loop)
            test_data_loader, test_data = dataset.data_parse(os.path.join(args_copy_loop.tfrecords, 'test_data.tfrecords'), type='test')
            if not os.path.exists(best_model):
                train_data_loader, val_data_loader, _ = dataset.data_parse(os.path.join(args_copy_loop.tfrecords, 'train_data.tfrecords'), type='train')
                #val_data_loader, _ = dataset.data_parse(os.path.join(args.tfrecords, 'val_data.tfrecords'), type='test')
                if is_tradition_alg(algorithm):
                    train_tradition_algs(algorithm, args_copy_loop, train_data_loader, best_model)
                else:
                    loss_min, f1_max, acc_max, es_count, epoch_max = 1000, 0, 0, 0, 0
                    method = RNNMethod(algorithm, args_copy_loop)
                    method.best_oa = 0
                    for epoch in range(args_copy_loop.iter_num):
                        #print('epoch: {}'.format(epoch))
                        method.train(train_data_loader, epoch)
                        loss_val, acc_val, f1_val = method.val(val_data_loader)
                        if epoch % 1000 == 0:
                            loss, acc, macro_f1 = method.val(test_data_loader)
                            print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                        if loss_val <= loss_min:
                            loss_min = loss_val
                            acc_max = acc_val
                            f1_max = f1_val
                            epoch_max = epoch
                            # saving model
                            torch.save(method.model.state_dict(), best_model)
                            print("Validate epoch: {} acc: {:>0.2f}%, loss: {:>8f}, macro-f1: {:>8f}, Saved PyTorch Model State to model.pth".format(epoch_max, acc_max, loss_min, f1_max))
                            loss, acc, macro_f1 = method.val(test_data_loader)
                            print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                    #torch.save(method.model.state_dict(), best_model)
                    print("best performance: epoch {}, loss: {:.6f}, acc: {:.2f}%,  f1: {:.6f}".format(epoch_max, loss_min, acc_max, f1_max))
                    #val_score_list.append([epoch_max, loss_min, acc_max, f1_max])

            print('{} loop: {}'.format(data_file, loop))
            if is_tradition_alg(algorithm):
                oa, aa, kappa, ac_list, loss, f1, pred, matrix_pro = test_tradition_algs(algorithm, args_copy_loop, test_data, best_model, test_data_loader)
            else:
                method = RNNMethod(algorithm, args_copy_loop)
                method.model.load_state_dict(torch.load(best_model, map_location='cpu'))
                oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = method.test(test_data_loader)
            test_score_list.append([algorithm, data_file, loop, oa, aa, kappa, ac_list, loss, f1])
            matrix_pro_list.append(matrix_pro)

    print('test_score_list:')
    df = pd.DataFrame(data=test_score_list, columns=['algorithm', 'data_file', 'loop', 'oa', 'aa', 'kappa', 'acc_list', 'loss', 'f1'])
    print(df)
    df.to_csv(os.path.join(args.result, data_type + '_' + str(file_no) + '_' + algorithm + '_test_scores.csv'), index=False)
    pickle.dump(matrix_pro_list, open(os.path.join(args.result, data_type + '_' + str(file_no) + '_' + algorithm + '_matrix_probabilities.pkl'), 'wb'))


def multi_process_kfold(algorithm, data_type='SWIR'):
    test_score_list = []
    matrix_pro_list = []
    matrix_list = []
    voting_test_score_list = []
    voting_matrix_pro_list = []
    total_test_time = datetime.now() - datetime.now()
    file_no = config.file_no
    if data_type == 'SWIR':
        data_file_list = ['SWIR_100051', 'SWIR_100052', 'SWIR_100053']
    else:
        data_file_list = ['VNIR_100051', 'VNIR_100052', 'VNIR_100053']
    args_copy = copy.deepcopy(args)

    for data_file in data_file_list:
        pred_list = []
        args_copy.data_file = data_file
        args_copy.model = os.path.join(args.model, data_file)
        args_copy.log = os.path.join(args.log, data_file)
        args_copy.result = os.path.join(args.result, data_file)
        args_copy.tfrecords = os.path.join(args.tfrecords, data_file)

        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        print('{}: file_no {}'.format(datetime.now(), file_no))
        args_copy.id = str(file_no)
        args_copy.result = os.path.join(args_copy.result, args_copy.id)
        args_copy.log = os.path.join(args_copy.log, args_copy.id)
        args_copy.model = os.path.join(args_copy.model, args_copy.id)
        args_copy.tfrecords = os.path.join(args_copy.tfrecords, args_copy.id)
        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        dataset = Data(args_copy)
        if not os.path.exists(os.path.join(args_copy.tfrecords, 'train_data.tfrecords')):
            dataset.read_data()
        if not os.path.exists(os.path.join(args_copy.tfrecords, 'kfold_train_9.pkl')):
            dataset.data_parse(os.path.join(args_copy.tfrecords, 'train_data.tfrecords'), type='train')
        for loop in range(2):
            best_model = os.path.join(args.model, data_file_list[0], str(file_no)) + '/model_{}_kfold_{}.pth'.format(args_copy.method, loop)
            test_data_loader, test_data = dataset.data_parse(os.path.join(args_copy.tfrecords, 'test_data.tfrecords'), type='test')

            pre_models = []
            if args.method == FMC_Str:
                config.loss_min = 0.00001
                pretrained_models = ['2D_CNN', 'RNN', '3D_CNN']
                pretrained_models = ['ResNet34_2D', 'RNN', 'ResNet34_3D']
                for index in range(len(pretrained_models)):
                    model_i = RNNMethod(pretrained_models[index], args_copy)
                    model_pth = os.path.join(args.model, data_file_list[0],
                                             str(file_no)) + '/model_{}_kfold_{}.pth'.format(pretrained_models[index],
                                                                                             loop)
                    model_i.model.load_state_dict(torch.load(model_pth, map_location='cpu'))
                    pre_models.append(model_i)

            if not os.path.exists(best_model):
                train_data_loader, _ = dataset.data_parse(os.path.join(args_copy.tfrecords, 'kfold_train_' + str(loop)) + '.pkl', type='kfold')
                val_data_loader, _ = dataset.data_parse(os.path.join(args_copy.tfrecords, 'kfold_val_' + str(loop)) + '.pkl', type='kfold')
                if is_tradition_alg(algorithm):
                    train_tradition_algs(algorithm, args_copy, train_data_loader, best_model, pre_models)
                else:
                    loss_min, f1_max, acc_max, es_count, epoch_max = 1000, 0, 0, 0, 0
                    method = RNNMethod(algorithm, args_copy)
                    method.best_oa = 0
                    for epoch in range(args_copy.iter_num):
                        #print('epoch: {}'.format(epoch))
                        method.train(train_data_loader, epoch, pre_models)
                        loss_val, acc_val, f1_val = method.val(val_data_loader, pre_models)
                        if epoch % 1000 == 0:
                            loss, acc, macro_f1 = method.val(test_data_loader, pre_models)
                            print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")

                        if loss_val <= loss_min:
                            loss_min = loss_val
                            acc_max = acc_val
                            f1_max = f1_val
                            epoch_max = epoch
                            # saving model
                            torch.save(method.model.state_dict(), best_model)
                            print("Validate epoch: {} acc: {:>0.2f}%, loss: {:>8f}, macro-f1: {:>8f}, Saved PyTorch Model State to model.pth".format(epoch_max, acc_max, loss_min, f1_max))
                            loss, acc, macro_f1 = method.val(test_data_loader, pre_models)
                            print(f"Test acc: {acc :>0.2f}%, loss: {loss:>8f}, macro-f1: {macro_f1:>8f}")
                            if loss_val <= config.loss_min:
                                break

                    #torch.save(method.model.state_dict(), best_model)
                    print("best performance: epoch {}, loss: {:.6f}, acc: {:.2f}%,  f1: {:.6f}".format(epoch_max, loss_min, acc_max, f1_max))
                    #val_score_list.append([epoch_max, loss_min, acc_max, f1_max])

            print('{} loop: {}'.format(data_file, loop))
            if is_tradition_alg(algorithm):
                oa, aa, kappa, ac_list, loss, f1, pred, matrix_pro, matrix, labels = test_tradition_algs(algorithm, args_copy, test_data, best_model, test_data_loader, pre_models)
            else:
                method = RNNMethod(algorithm, args_copy)
                method.model.load_state_dict(torch.load(best_model, map_location='cpu'))
                oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = method.test(test_data_loader, pre_models)
            test_score_list.append([algorithm, data_file, loop, oa, aa, kappa, ac_list, loss, f1])
            matrix_pro_list.append(matrix_pro)
            if is_tradition_alg(algorithm):
                matrix_list.append(matrix)
            pred_list.append(pred)
        pred_arr = np.array(pred_list)
        pred_arr = np.sum(pred_arr, axis=0)
        oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = model_i.metrics(pred_arr, labels)
        voting_test_score_list.append(['voting_models_fmc_lda', data_file, oa, aa, kappa, ac_list, loss, f1])
        voting_matrix_pro_list.append(matrix_pro)

    print('test_score_list:')
    df = pd.DataFrame(data=test_score_list, columns=['algorithm', 'data_file', 'loop', 'oa', 'aa', 'kappa', 'acc_list', 'loss', 'f1'])
    print(df)
    df.to_csv(os.path.join(args.result, data_type + '_' + str(file_no) + '_' + algorithm + '_kfold_test_scores.csv'), index=False)
    pickle.dump(matrix_pro_list, open(os.path.join(args.result, data_type + '_' + str(file_no) + '_' + algorithm + '_kfold_matrix_probabilities.pkl'), 'wb'))
    if is_tradition_alg(algorithm):
        pickle.dump(matrix_list, open(os.path.join(args.result, data_type + '_' + str(file_no) + '_' + algorithm + '_kfold_matrix.pkl'), 'wb'))


def multi_process_kfold_softvoting(data_type='SWIR'):
    test_score_list = []
    matrix_pro_list = []
    voting_test_score_list = []
    voting_matrix_pro_list = []
    total_test_time = datetime.now() - datetime.now()
    file_no = config.file_no
    if data_type == 'SWIR':
        data_file_list = ['SWIR_100051', 'SWIR_100052', 'SWIR_100053']
    else:
        data_file_list = ['VNIR_100051', 'VNIR_100052', 'VNIR_100053']
    args_copy = copy.deepcopy(args)

    for data_file in data_file_list:
        args_copy.data_file = data_file
        args_copy.model = os.path.join(args.model, data_file)
        args_copy.log = os.path.join(args.log, data_file)
        args_copy.result = os.path.join(args.result, data_file)
        args_copy.tfrecords = os.path.join(args.tfrecords, data_file)

        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        print('{}: file_no {}'.format(datetime.now(), file_no))
        args_copy.id = str(file_no)
        args_copy.result = os.path.join(args_copy.result, args_copy.id)
        args_copy.log = os.path.join(args_copy.log, args_copy.id)
        args_copy.model = os.path.join(args_copy.model, args_copy.id)
        args_copy.tfrecords = os.path.join(args_copy.tfrecords, args_copy.id)
        if not os.path.exists(args_copy.model):
            os.mkdir(args_copy.model)
        if not os.path.exists(args_copy.log):
            os.mkdir(args_copy.log)
        if not os.path.exists(args_copy.result):
            os.mkdir(args_copy.result)
        if not os.path.exists(args_copy.tfrecords):
            os.mkdir(args_copy.tfrecords)

        dataset = Data(args_copy)
        models_names = ['2D_CNN', 'RNN', '3D_CNN']
        models_weights = [0.2, 0.4, 0.4]
        test_data_loader, test_data = dataset.data_parse(os.path.join(args_copy.tfrecords, 'test_data.tfrecords'), type='test')
        pred_list = []
        for loop in range(2):
            voting_models = []
            for index in range(len(models_names)):
                model_i = RNNMethod(models_names[index], args_copy)
                model_pth = os.path.join(args.model, data_file_list[0], str(file_no)) + '/model_{}_kfold_{}.pth'.format(models_names[index], loop)
                model_i.model.load_state_dict(torch.load(model_pth, map_location='cpu'))
                voting_models.append(model_i)
            oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = model_i.voting_test(test_data_loader, models_weights, voting_models)
            test_score_list.append(['voting_models', data_file, loop, oa, aa, kappa, ac_list, loss, f1])
            matrix_pro_list.append(matrix_pro)
            pred_list.append(pred)
        pred_arr = np.array(pred_list)
        pred_arr = np.sum(pred_arr, axis=0)

        oa, aa, kappa, ac_list, loss, f1, pred, labels, matrix_pro = model_i.metrics(pred_arr, labels)
        voting_test_score_list.append(['voting_models', data_file, oa, aa, kappa, ac_list, loss, f1])
        voting_matrix_pro_list.append(matrix_pro)
    print('test_score_list:')
    df = pd.DataFrame(data=test_score_list,
                      columns=['algorithm', 'data_file', 'loop', 'oa', 'aa', 'kappa', 'acc_list', 'loss', 'f1'])
    print(df)
    df.to_csv(os.path.join(args.result, data_type + '_' + str(file_no) + '_softvoting_kfold_test_scores.csv'), index=False)
    pickle.dump(matrix_pro_list, open(
        os.path.join(args.result, data_type + '_' + str(file_no) + '_softvoting_kfold_matrix_probabilities.pkl'), 'wb'))

    print('softvoting_test_score_list:')
    df = pd.DataFrame(data=voting_test_score_list,
                      columns=['algorithm', 'data_file', 'oa', 'aa', 'kappa', 'acc_list', 'loss', 'f1'])
    print(df)
    df.to_csv(os.path.join(args.result, data_type + '_' + str(file_no) + '_softvoting_test_scores.csv'), index=False)
    pickle.dump(voting_matrix_pro_list, open(
        os.path.join(args.result, data_type + '_' + str(file_no) + '_softvoting_matrix_probabilities.pkl'), 'wb'))

if __name__ == "__main__":
    #single_process(args.method)
    #multi_process_kfold_softvoting(data_type='SWIR')
    multi_process_kfold(args.method, data_type='SWIR')