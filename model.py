import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2
# from MobileNetV2 import mobilenet_v2


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, device, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        # self.lstm_layers = lstm_layers
        # self.lstm_hidden = lstm_hidden
        # self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_logits = 5  # one for each club
        self.device = device

        # net = mobilenet_v2(pretrained=True) # I added this line!!!

        net = MobileNetV2(n_class=self.num_logits)
        state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar')
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])

        self.cnn_out_lin = nn.Linear(int(1280*width_mult if width_mult > 1.0 else 1280), self.num_logits)


        # self.rnn = nn.LSTM(int(1280*width_mult if width_mult > 1.0 else 1280),
        #                    self.lstm_hidden, self.lstm_layers,
        #                    batch_first=True, bidirectional=bidirectional)
        # if self.bidirectional:
        #     self.lin = nn.Linear(2*self.lstm_hidden, self.num_logits)
        # else:
        #     self.lin = nn.Linear(self.lstm_hidden, self.num_logits)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        # if self.bidirectional:
        #     return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True),
        #             Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True))
        # else:
        #     return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True),
        #             Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True))
        return

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        # self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        out = self.cnn_out_lin(c_out)

        # LSTM forward
        # r_in = c_out.view(batch_size, timesteps, -1)
        # r_out, states = self.rnn(r_in, self.hidden)
        # out = self.lin(r_out)
        out = out.view(batch_size*timesteps, self.num_logits)

        return out



