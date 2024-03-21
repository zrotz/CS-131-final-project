import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2
# from MobileNetV2 import mobilenet_v2


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, device, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.dropout = dropout
        self.num_logits = 5  # one for each club
        self.device = device

        net = MobileNetV2(n_class=self.num_logits)
        state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar')
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])

        self.cnn_out_lin = nn.Linear(int(1280*width_mult if width_mult > 1.0 else 1280), self.num_logits)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size())

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        out = self.cnn_out_lin(c_out)

        out = out.view(batch_size*timesteps, self.num_logits)

        return out



