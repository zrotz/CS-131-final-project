from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
from tqdm import tqdm


def eval(model, split, seq_length, device, disp):
    dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    num_correct = 0
    total = 0

    for i, sample in tqdm(enumerate(data_loader)):
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        probs = np.asarray(probs)
        labels = labels.squeeze().cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        c = (predictions == labels)
        num_correct += np.sum(c)
        total += labels.shape[0]

    # PCE = np.mean(correct)
    PCE = num_correct / total
    print(PCE)
    return PCE


if __name__ == '__main__':

    # split = 1
    seq_length = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False)

    folder = 'models/cnn_only'
    save_dict = torch.load(f'{folder}/swingnet_2000.pth.tar', map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()

    PCE_list = []
    for i in range(1, 5):
        PCE_list.append(eval(model, i, seq_length, device, True))

    PCE = sum(PCE_list) / len(PCE_list)
    print('Average PCE: {}'.format(PCE))


