from mockingjay.nn_mockingjay import MOCKINGJAY
from downstream.model import example_classifier
from downstream.solver import get_mockingjay_optimizer
import torch
import torch.nn as nn
import os
from torch.utils import data
import numpy as np
from tqdm import tqdm

# setup the mockingjay models



class Dataset(data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = os.listdir(folder)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.folder, file)
        data = np.load(path)
        label = int(file[:-4].split('-')[-1])
        return torch.from_numpy(data), torch.tensor(label)


class Model(nn.Module):
    def __init__(self, options):
        super(Model, self).__init__()
        self.model = MOCKINGJAY(options=options, inp_dim=160)
        self.classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=133)

    def forward(self, inp, target):
        inp = inp.permute(1, 0, 2)
        hidden = self.model(inp) # returns: (time_step, batch_size, hidden_size)
        hidden = hidden.permute(1, 0, 2)
        loss = self.classifier(hidden, target)
        return loss


def train(loader, epochs, options):
    # setup your downstream class model

    # construct the Mockingjay optimizer
    model = Model(options).cuda()
    params = list(model.named_parameters())
    optimizer = get_mockingjay_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

    tot_loss = 0.
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for batch in tqdm(loader):
            inp, target = batch
            inp, target = inp.cuda(), target.cuda()
            loss = train_one_step(inp, target, model, optimizer)
            tot_loss += loss
        print(f'train_mean_loss: {tot_loss / len(loader)}')
        torch.save(f'ckpt/epoch{epoch:.2d}.ckpt')



def train_one_step(inp, target, model, optimizer):
    loss = model(inp, target) 

    loss.backward()
    optimizer.step()
    return loss.item()

# forward
# example_inputs = torch.zeros(1200, 3, 160) # A batch of spectrograms: (time_step, batch_size, dimension)

def main(folder_path, options):
    ds = Dataset(folder_path)
    loader = data.DataLoader(ds, num_workers=10, shuffle=True)
    train(loader, epochs=10, options=options)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    DATASET_PATH = 'libri_mel160/train-clean-100'
    options = {
        'ckpt_file' : 'result/mockingjay_libri_sd1337_MelBase/mockingjay-500000.ckpt',
        'load_pretrain' : 'True',
        'no_grad' : 'False',
        'dropout' : 'default'
    }
    main(DATASET_PATH, options)