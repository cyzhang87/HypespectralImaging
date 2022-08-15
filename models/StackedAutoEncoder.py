import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCELoss
from torch import optim

class AutoEncoder(nn.Module):
    def __init__(self, nIn, nOut):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 28*28),
                                     nn.Tanh())

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class AutoEncoderLayer(nn.Module):
    """
    fully-connected linear layers for stacked autoencoders.
    This module can automatically be trained when training each layer is enabled
    Yes, this is much like the simplest auto-encoder
    """

    def __init__(self, input_dim=None, output_dim=None, SelfTraining=False):
        super(AutoEncoderLayer, self).__init__()
        # if input_dim is None or output_dim is None:
        #     raise ValueError
        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training_self = SelfTraining  # 指示是否进行逐层预训练,还是训练整个网络
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=True),
            nn.Sigmoid()  # 统一使用Sigmoid激活
        )
        self.decoder = nn.Sequential(  # 此处decoder不使用encoder的转置, 并使用Sigmoid进行激活.
            nn.Linear(self.out_features, self.in_features, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        if self.is_training_self:
            return self.decoder(out)
        else:
            return out

    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter
    def is_training_layer(self, other: bool):
        self.is_training_self = other



class StackedAutoEncoder(nn.Module):
    """
    Construct the whole network with layers_list
    """

    def __init__(self, layers_list=None):
        super(StackedAutoEncoder, self).__init__()
        self.layers_list = layers_list
        self.initialize()
        self.encoder_1 = self.layers_list[0]
        self.encoder_2 = self.layers_list[1]
        self.encoder_3 = self.layers_list[2]
        self.encoder_4 = self.layers_list[3]

    def initialize(self):
        for layer in self.layers_list:
            # assert isinstance(layer, AutoEncoderLayer)
            layer.is_training_layer = False
            # for param in layer.parameters():
            #     param.requires_grad = True

    def forward(self, x):
        out = x
        # for layer in self.layers_list:
        #     out = layer(out)
        out = self.encoder_1(out)
        out = self.encoder_2(out)
        out = self.encoder_3(out)
        out = self.encoder_4(out)
        return out

class SAEModel(nn.Module):
    def __init__(self):
        super(SAEModel, self).__init__()
        self.nun_layers = 5
        encoder_1 = AutoEncoderLayer(5248, 1024, SelfTraining=True)
        encoder_2 = AutoEncoderLayer(1024, 256, SelfTraining=True)
        decoder_3 = AutoEncoderLayer(256, 1024, SelfTraining=True)
        decoder_4 = AutoEncoderLayer(1024, 5248, SelfTraining=True)
        self.layers_list = [encoder_1, encoder_2, decoder_3, decoder_4]
        self.model = StackedAutoEncoder(layers_list=self.layers_list)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_layers(self, train_loader, layer=None, epoch=None, validate=True):
        """
        逐层贪婪预训练 --当训练第i层时, 将i-1层冻结
        :param layers_list:
        :param layer:
        :param epoch:
        :return:
        """
        if torch.cuda.is_available():
            for model in layers_list:
                model.cuda()
        optimizer = optim.SGD(self.layers_list[layer].parameters(), lr=0.001)
        criterion = BCELoss()

        # train
        for epoch_index in range(epoch):
            sum_loss = 0.

            # 冻结当前层之前的所有层的参数  --第0层没有前置层
            if layer != 0:
                for index in range(layer):
                    layers_list[index].lock_grad()
                    layers_list[index].is_training_layer = False  # 除了冻结参数,也要设置冻结层的输出返回方式

            for batch_index, (train_data, _) in enumerate(train_loader):
                # 生成输入数据
                if torch.cuda.is_available():
                    train_data = train_data.cuda()  # 注意Tensor放到GPU上的操作方式,和model不同
                out = train_data.view(train_data.size(0), -1)

                # 对前(layer-1)冻结了的层进行前向计算
                if layer != 0:
                    for l in range(layer):
                        out = layers_list[l](out)

                # 训练第layer层
                pred = layers_list[layer](out)

                optimizer.zero_grad()
                loss = criterion(pred, out)
                sum_loss += loss
                loss.backward()
                optimizer.step()
                if (batch_index + 1) % 10 == 0:
                    print("Train Layer: {}, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                        layer, (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
                    ))

            if validate:
                pass


    def train_whole(self, model=None, epoch=50, validate=True):
        print(">> start training whole model")
        if torch.cuda.is_available():
            model.cuda()

        # 解锁因预训练单层而冻结的参数
        for param in model.parameters():
            param.require_grad = True

        train_loader, test_loader = get_mnist_loader(batch_size=batch_size, shuffle=shuffle)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        # criterion = BCELoss()
        criterion = torch.nn.MSELoss()

        # 生成/保存原始test图片 --取一个batch_size
        test_data, _ = next(iter(test_loader))
        torchvision.utils.save_image(test_data, './test_images/real_test_images.png')

        # train
        for epoch_index in range(epoch):
            sum_loss = 0.
            for batch_index, (train_data, _) in enumerate(train_loader):
                if torch.cuda.is_available():
                    train_data = train_data.cuda()
                x = train_data.view(train_data.size(0), -1)

                out = model(x)

                optimizer.zero_grad()
                loss = criterion(out, x)
                sum_loss += loss
                loss.backward()
                optimizer.step()

                if (batch_index + 1) % 10 == 0:
                    print("Train Whole, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                        (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
                    ))
                if batch_index == len(train_loader) - 1:
                    torchvision.utils.save_image(out.view(100, 1, 28, 28), "./test_images/out_{}_{}.png".format(epoch_index, batch_index))

            # 每个epoch验证一次
            if validate:
                if torch.cuda.is_available():
                    test_data = test_data.cuda()
                x = test_data.view(test_data.size(0), -1)
                out = model(x)
                loss = criterion(out, x)
                print("Test Epoch: {}/{}, Iter: {}/{}, test Loss: {}".format(
                    epoch_index + 1, epoch, (epoch_index + 1), len(test_loader), loss
                ))
                image_tensor = out.view(batch_size, 1, 28, 28)
                torchvision.utils.save_image(image_tensor, './test_images/test_image_epoch_{}.png'.format(epoch_index))
        print("<< end training whole model")







if __name__ == "__main__":
    # sae
    nun_layers = 5
    encoder_1 = AutoEncoderLayer(784, 256, SelfTraining=True)
    encoder_2 = AutoEncoderLayer(256, 64, SelfTraining=True)
    decoder_3 = AutoEncoderLayer(64, 256, SelfTraining=True)
    decoder_4 = AutoEncoderLayer(256, 784, SelfTraining=True)
    layers_list = [encoder_1, encoder_2, decoder_3, decoder_4]
    model = StackedAutoEncoder(layers_list=layers_list)
    input_var = Variable(torch.randn(100, 784))
    output = model(input_var)
    # end sae

    print(output.shape)