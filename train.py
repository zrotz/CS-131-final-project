from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from MobileNetV2 import MobileNetV2



if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    seq_length = 64
    bs = 22  # batch size
    k = 10  # frozen layers
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device = ", device)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.to(device)

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             drop_last=True)
    

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    # weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=None)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    folder = 'models/cnn_only'
    if not os.path.exists(folder):
        os.mkdir(folder)


    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            # print('labels = ', labels.size())
            logits = model(images)
            # print('logits = ', logits.size())
            labels = labels.view(bs*seq_length)
            # print('labels resized = ', labels.size())
            loss = criterion(logits, labels)
            # print('loss = ', loss)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                print("Saving model")
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, f'{folder}/swingnet_{i}.pth.tar')
            if i == iterations:
                break

