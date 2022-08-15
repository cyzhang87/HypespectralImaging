import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BidirectionalGRU(nn.Module):

    def __init__(self, nIn, nHidden, nOut, use_dropout=False):
        super(BidirectionalGRU, self).__init__()
        self.rnn1 = nn.GRU(nIn, nHidden, batch_first=True, bidirectional=True, dropout=0.5 if use_dropout else 0)
        self.fc = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, input):
        recurrent, (hidden, cell) = self.rnn1(input)
        rnn1_output = recurrent[:, -1, :]  # get the last sequence
        output = self.fc(rnn1_output)
        return output


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, use_dropout=False):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, batch_first=True, bidirectional=True, dropout=0.5 if use_dropout else 0)
        self.fc = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        rnn1_output = recurrent[:, -1, :]  # get the last sequence
        output = self.fc(rnn1_output)

        return output


class Conv1d(nn.Module):

    def __init__(self, nIn, nHidden, nkernel, nOut, nSeq_len, use_dropout=False):
        super(Conv1d, self).__init__()
        out_ch_1 = 32
        out_ch_2 = 64
        stride = 2
        self.conv1d_1 = nn.Conv1d(in_channels=nIn, out_channels=out_ch_1, kernel_size=nkernel, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=out_ch_1, out_channels=out_ch_2, kernel_size=nkernel, padding=2)
        self.active = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=stride)
        #self.maxpool = self.soft_pool1d(kernel_size=2, stride=stride)
        self.dropout = nn.Dropout(p=0.5, inplace=use_dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(out_ch_2 * (nSeq_len) / stride), nOut)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        output = self.conv1d_1(input)
        output = self.active(output)
        output = self.conv1d_2(output)
        output = self.active(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        output = self.flatten(output)
        output = self.fc(output)

        return output


class Conv2d(nn.Module):

    def __init__(self, nIn, nkernel, out_ch):
        super(Conv2d, self).__init__()
        self.conv2d_00 = nn.Conv2d(in_channels=nIn, out_channels=out_ch, kernel_size=nkernel, padding='same')
        self.conv2d_01 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 1), padding='same')
        self.conv2d_02 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=nkernel, padding='same')
        self.conv2d_10 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch * 2, kernel_size=nkernel, padding='same')
        self.conv2d_11 = nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=(1, 1), padding='same')
        self.conv2d_12 = nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=nkernel, padding='same')
        self.conv2d_20 = nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 4, kernel_size=nkernel, padding='same')
        self.conv2d_21 = nn.Conv2d(in_channels=out_ch * 4, out_channels=out_ch * 4, kernel_size=(1, 1), padding='same')
        self.conv2d_22 = nn.Conv2d(in_channels=out_ch * 4, out_channels=out_ch * 4, kernel_size=nkernel, padding='valid')
        self.conv2d_30 = nn.Conv2d(in_channels=out_ch * 4, out_channels=out_ch * 8, kernel_size=(1, 1))

        self.active = nn.ReLU()
        self.norm_00 = nn.BatchNorm2d(out_ch)
        self.norm_01 = nn.BatchNorm2d(out_ch)
        self.norm_02 = nn.BatchNorm2d(out_ch)
        self.norm_10 = nn.BatchNorm2d(out_ch * 2)
        self.norm_11 = nn.BatchNorm2d(out_ch * 2)
        self.norm_12 = nn.BatchNorm2d(out_ch * 2)
        self.norm_20 = nn.BatchNorm2d(out_ch * 4)
        self.norm_21 = nn.BatchNorm2d(out_ch * 4)
        self.norm_22 = nn.BatchNorm2d(out_ch * 4)
        self.flatten = nn.Flatten()

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        output = self.conv2d_00(input)
        output = self.norm_00(output)
        output = self.active(output)

        output = self.conv2d_01(output)
        output = self.norm_01(output)
        output = self.active(output)

        output = self.conv2d_02(output)
        output = self.norm_02(output)
        output = self.active(output)

        output = self.conv2d_10(output)
        output = self.norm_10(output)
        output = self.active(output)

        output = self.conv2d_11(output)
        output = self.norm_11(output)
        output = self.active(output)

        output = self.conv2d_12(output)
        output = self.norm_12(output)
        output = self.active(output)

        output = self.conv2d_20(output)
        output = self.norm_20(output)
        output = self.active(output)

        output = self.conv2d_21(output)
        output = self.norm_21(output)
        output = self.active(output)

        output = self.conv2d_22(output)
        output = self.norm_22(output)
        output = self.active(output)

        output = self.conv2d_30(output)
        output = self.flatten(output) # 128*512
        return output

class Conv2dModel(nn.Module):

    def __init__(self, nIn, nkernel, nOut):
        super(Conv2dModel, self).__init__()
        out_ch = 64
        self.conv2d = Conv2d(nIn, nkernel, out_ch)
        self.fc = nn.Linear(out_ch * 8, nOut)

    def forward(self, input):
        output = self.conv2d(input)
        feature_output = output
        output = self.fc(output)
        FEATURE_OUT = True
        if FEATURE_OUT:
            return output, feature_output
        else:
            return output


class Conv3d(nn.Module):

    def __init__(self, nIn, nkernel, out_ch, mp_ks_1, mp_ks_2, stride):
        super(Conv3d, self).__init__()
        self.conv3d_00 = nn.Conv3d(in_channels=nIn, out_channels=out_ch, kernel_size=nkernel, padding='same')
        self.conv3d_10 = nn.Conv3d(in_channels=out_ch, out_channels=out_ch * 2, kernel_size=(1, 1, 1), padding='same')

        self.maxpool = nn.MaxPool3d(kernel_size=mp_ks_1, stride=stride)
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(1, 1, mp_ks_2), stride=(1, 1, stride))
        self.active = nn.ReLU()
        self.norm_00 = nn.BatchNorm3d(out_ch)
        self.norm_10 = nn.BatchNorm3d(out_ch * 2)


    def forward(self, input):
        input = input[:, None, :, :, :]  # add a dim 128*3*3*272 -> 128*1*3*3*272
        #input = torch.tensor(np.expand_dims(input.cpu(), axis=-1)).to(device)
        #input = input.permute(0, 4, 1, 2, 3) # 128*1*3*3*272, batch_size:128
        output = self.conv3d_00(input) # 128*64*3*3*272
        output = self.norm_00(output) # 128*64*3*3*272
        output = self.active(output) # 128*64*3*3*272
        output = self.maxpool(output) # 128*64*3*3*135

        output = self.conv3d_10(output) # 128*128*1*1*135
        output = self.norm_10(output)
        output = self.active(output)
        output = self.maxpool_2(output) # 128*128*1*1*67
        output = self.maxpool_2(output) # 128*128*1*1*33
        #output = self.conv3d_30(output)
        return output


class Conv3dModel(nn.Module):

    def __init__(self, nIn, ndim, nkernel, nOut):
        super(Conv3dModel, self).__init__()
        out_ch = 64
        mp_ks_1 = 3
        mp_ks_2 = 2
        stride = 2
        self.conv3d = Conv3d(nIn, nkernel, out_ch, mp_ks_1, mp_ks_2, stride)
        mp_out = int((ndim - mp_ks_1) / stride + 1)
        mp_out = int((mp_out - mp_ks_2) / stride + 1)
        mp_out = int((mp_out - mp_ks_2) / stride + 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_ch * 2 * mp_out, nOut)

    def forward(self, input):
        output = self.conv3d(input) # 128*128*1*1*67
        output = self.flatten(output)  # 128*8576
        feature_output = output
        output = self.fc(output)
        FEATURE_OUT = True
        if FEATURE_OUT:
            return output, feature_output
        else:
            return output

class MLayerBiLSTM(nn.Module):

    def __init__(self, nIn, nHidden, use_dropout=False):
        super(MLayerBiLSTM, self).__init__()
        self.rnn1 = nn.LSTM(nIn, nHidden, batch_first=True, bidirectional=True, dropout=0.5 if use_dropout else 0)
        self.rnn2 = nn.LSTM(nHidden * 2, nHidden, batch_first=True, bidirectional=True,
                            dropout=0.5 if use_dropout else 0)

    def attention(self, lstm_output, final_state):
        # lstm_output = lstm_output.permute(0, 2, 1)
        merged_state = torch.cat([s for s in final_state], 1) #128*256
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, input):
        recurrent1, (hidden, cell) = self.rnn1(input) # input:128*272*1 recurrent1:128*272*256
        recurrent2, (hidden, cell) = self.rnn2(recurrent1)
        #attention_output = self.attention(recurrent2, hidden) #recurrent2: 128*67*256, hidden: 2*128*128
        #rnn1_output = recurrent1[:, -1, :]  # get the last sequence
        rnn2_output = recurrent2[:, -1, :]
        #output = torch.cat([attention_output, rnn2_output], 1)
        return rnn2_output


class MLayerBiLSTMModel(nn.Module):

    def __init__(self, nIn, nOut, use_dropout=False):
        super(MLayerBiLSTMModel, self).__init__()
        nHidden = 64 * 2
        self.rnn = MLayerBiLSTM(nIn, nHidden, use_dropout=use_dropout)
        self.fc = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        input = torch.mean(torch.mean(input, axis=1), axis=1, keepdims=True).permute(0, 2, 1) #128*272*1
        output = self.rnn(input) # 128*512
        feature_output = output
        output = self.fc(output)
        FEATURE_OUT = True
        if FEATURE_OUT:
            return output, feature_output
        else:
            return output


class MLayerBiGRU(nn.Module):

    def __init__(self, nIn, nHidden, use_dropout=False):
        super(MLayerBiGRU, self).__init__()
        self.rnn1 = nn.GRU(nIn, nHidden, batch_first=True, bidirectional=True, dropout=0.5 if use_dropout else 0)
        self.rnn2 = nn.GRU(nHidden * 2, nHidden, batch_first=True, bidirectional=True, dropout=0.5 if use_dropout else 0)

    def attention(self, lstm_output, final_state):
        # lstm_output = lstm_output.permute(0, 2, 1)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, input):
        recurrent1, hidden = self.rnn1(input) # recurrent1: 128*272*256
        recurrent2, hidden = self.rnn2(recurrent1)  # recurrent2: 128*272*256, hidden: 2*128*128
        attention_output = self.attention(recurrent2, hidden)
        rnn1_output = recurrent1[:, -1, :]  # get the last sequence
        rnn2_output = recurrent2[:, -1, :] # 128*256
        output = torch.cat([attention_output, rnn2_output], 1) # 128*512
        return output


class Conv3d_RNN(nn.Module):

    def __init__(self, nIn, nkernel, nOut, rnn_type='lstm', use_dropout=True):
        super(Conv3d_RNN, self).__init__()
        # conv3d
        out_ch = 64
        mp_ks_1 = 3
        mp_ks_2 = 2
        stride = 2
        self.conv3d = Conv3d(nIn, nkernel, out_ch, mp_ks_1, mp_ks_2, stride)

        # BiRNN
        nHidden = out_ch * 2
        if rnn_type == 'lstm':
            self.rnn = MLayerBiLSTM(out_ch * 2, nHidden, use_dropout=use_dropout)
        elif rnn_type == 'gru':
            self.rnn = MLayerBiGRU(out_ch * 2, nHidden, use_dropout=use_dropout)
        self.fc = nn.Linear(nHidden * 4, nOut)

    def forward(self, input):
        output = self.conv3d(input) # 128*128*1*1*67
        output = output[:,:,0,0,:] # 128*128*67
        output = output.permute(0, 2, 1) # 128*67*128
        output = self.rnn(output) # 128*512
        feature_output = output
        output = self.fc(output)
        FEATURE_OUT = True
        if FEATURE_OUT:
            return output, feature_output
        else:
            return output


class JointDeepModel(nn.Module):

    def __init__(self, nIn_2d, nIn_3d, nkernel_2d, nkernel_3d, nOut, rnn_type='lstm', use_dropout=True):
        super(JointDeepModel, self).__init__()
        # conv2d
        out_ch_2d = 64
        self.conv2d = Conv2d(nIn_2d, nkernel_2d, out_ch_2d)

        # BiRNN
        nHidden = 64 * 2
        if rnn_type == 'lstm':
            self.rnn = MLayerBiLSTM(1, nHidden, use_dropout=use_dropout)
        elif rnn_type == 'gru':
            self.rnn = MLayerBiGRU(1, nHidden, use_dropout=use_dropout)

        # conv3d
        out_ch_3d = 32
        mp_ks_1 = 3
        mp_ks_2 = 2
        stride = 2
        mp_out = int((nIn_2d - mp_ks_1) / stride + 1)
        mp_out = int((mp_out - mp_ks_2) / stride + 1)
        mp_out = int((mp_out - mp_ks_2) / stride + 1)
        self.conv3d = Conv3d(nIn_3d, nkernel_3d, out_ch_3d, mp_ks_1, mp_ks_2, stride)

        # classifier
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_ch_2d * 8 + nHidden * 4 + out_ch_3d * 2 * mp_out, nOut)

    def forward(self, input):
        output_2d = self.conv2d(input) # 128*512

        output_rnn = torch.mean(torch.mean(input, axis=1), axis=1, keepdims=True).permute(0, 2, 1)
        output_rnn = self.rnn(output_rnn) # 128*512

        output_3d = self.conv3d(input)  # 128*64*1*1*33
        output_3d = output_3d[:, :, 0, 0, :]  # 128*64*33
        output_3d = self.flatten(output_3d) # 128*21128

        output = torch.cat([output_2d, output_rnn, output_3d], 1) #128*3136
        feature_output = output
        output = self.fc(output)
        FEATURE_OUT = True
        if FEATURE_OUT:
            return output, feature_output
        else:
            return output


class FMC(nn.Module):

    def __init__(self, nIn_2d, nOut):
        super(FMC, self).__init__()
        # conv2d
        out_ch_2d = 64

        # BiRNN
        nHidden = 64 * 2

        # conv3d
        out_ch_3d = 64
        mp_ks_1 = 3
        mp_ks_2 = 2
        stride = 2
        mp_out = int((nIn_2d - mp_ks_1) / stride + 1)
        mp_out = int((mp_out - mp_ks_2) / stride + 1)
        mp_out = int((mp_out - mp_ks_2) / stride + 1)

        # classifier
        #self.fc = nn.Linear(out_ch_2d * 8 + nHidden * 4 + out_ch_3d * 2 * mp_out, nOut)
        """
        self.classifier = nn.Sequential(
            nn.Linear(out_ch_2d * 8 + nHidden * 2 + out_ch_3d * 2 * mp_out, nOut), # 5248 --> 1024
            nn.ReLU(),
            #nn.Linear(1024, nOut),
            #nn.ReLU(),
            #nn.Linear(256, 64, bias=True),
            #nn.Sigmoid(),
            #nn.Linear(64, nOut, bias=True),
            #nn.ReLU()
        )"""
        self.classifier = nn.Sequential(
            nn.Linear(512 + nHidden * 2 + 512, nOut), # 1280
            nn.ReLU())


    def forward(self, input):
        output = self.classifier(input)
        FEATURE_OUT = True
        if FEATURE_OUT:
            return output, input
        else:
            return output



from torch.autograd import Variable
def get_model(**kwargs):
    """
    Returns the model.
    """
    model = Conv3d(**kwargs)
    return model



if __name__ == "__main__":
    model = get_model(nIn=1, ndim=273, nkernel=(3,3,3), nOut=6)
    #model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(100, 3, 3, 273))
    output, _ = model(input_var)
    print(output.shape)