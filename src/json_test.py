import json, os , time, copy, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths


class joint_dataset(Dataset):
    def __init__(self, root, print_path=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.X = []
        self.Y = []
        folder_paths = get_path(root)
        for path in folder_paths:
            score_json_paths = get_path(path)
            for json_path in score_json_paths:
                if print_path:
                    print(json_path)
                with open(json_path, 'r') as score_json:
                    frame_dict = json.load(score_json)
                    for i in range(len(frame_dict['frames']) - 3):
                        one = np.array(frame_dict['frames'][0 + i]['joint'], dtype=int).reshape((68))
                        # two = np.array(frame_dict['frames'][1 + i]['joint'], dtype=int).reshape((1, 68))
                        # three = np.array(frame_dict['frames'][2 + i]['joint'], dtype=int).reshape((1, 68))
                        # four = np.array(frame_dict['frames'][3 + i]['joint'], dtype=int).reshape((1, 68))
                        # print(np.concatenate([one, two, three, four]).shape)
                        self.X.append(one)
                        self.Y.append(frame_dict['frames'][1 + i]['label'])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float)

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        return self.X[index], self.Y[index]

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.X)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = joint_dataset('E:/labeled_data', False)
# dataloaders = {x: DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2) for x in ['train', 'val']}
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, :34] = torch.sin(position*div_term)
        pe[:, 0, 34:] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, window_size: int, d_model: int = 68, nhead: int = 2, d_hid: int = 512,
                 nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, window_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(self.window_size*self.d_model, 3)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # print('o: ',output.shape)
        output = output.reshape(-1, self.window_size*self.d_model)
        output = self.decoder(output)
        return output

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloader:
            # inputs = torch.permute(inputs, (1,0,2))
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # print(outputs.shape, labels.shape, preds.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# model = TransformerModel(1)

model = nn.Sequential(  nn.BatchNorm1d(68),
                        nn.Linear(68,256),
                        nn.ReLU(),
                        nn.Linear(256,128),
                        nn.ReLU(),
                        nn.Linear(128,3),).to(device)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)