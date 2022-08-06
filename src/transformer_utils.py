import torch
import torch.nn as nn
import math
import numpy as np
import json
import os
import torch.nn.functional as F


class coordinateEmbedding(nn.Module):
    def __init__(self, in_channels: int, emb_size: int):
        super().__init__()
        half_emb = int(emb_size / 2)
        self.projection1 = nn.Linear(in_channels, half_emb)
        self.projection1_2 = nn.Linear(half_emb, half_emb)
        self.projection2 = nn.Linear(in_channels, half_emb)
        self.projection2_2 = nn.Linear(half_emb, half_emb)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        p1 = x.select(2, 0)
        p2 = x.select(2, 1)
        p1 = self.projection1(p1)
        p2 = self.projection2(p2)
        p1 = self.projection1_2(p1)
        p2 = self.projection2_2(p2)
        projected = torch.cat((p1, p2), 2)
        #         print(projected.shape)
        return projected


class classEmbedding(nn.Module):
    def __init__(self, num_classes: int, emb_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(num_classes, emb_size, kernel_size=(1, 1), stride=1),
        )

    def forward(self, x):
        x = F.one_hot(x, num_classes=5)  # batch, token_num, emb_size
        x = x.to(torch.float32)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.projection(x).squeeze(-1)
        # batch, emb_size, window+2
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5  10,1
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Optimus_Prime(nn.Module):
    def __init__(
            self,
            num_tokens,
            dim_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward=128,
            dropout_p=0,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=600)

        self.xy_embedding = coordinateEmbedding(in_channels=24, emb_size=dim_model)

        self.tgt_tok_emb = classEmbedding(num_classes=5, emb_size=dim_model)

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            activation='gelu'
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None, memory_key_padding_mask=None):

        src = self.xy_embedding(src)  # batch, win+2, 2, 24
        # batch, 768, 12

        tgt = self.tgt_tok_emb(tgt)  # batch, win+1
        # batch, win+1, dim_model  3 SOS 4 EOS 0,1,2

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt,
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask,
                                           memory_key_padding_mask=memory_key_padding_mask
                                           )
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_tgt_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        #         print((matrix == pad_token).shape)
        return (matrix == pad_token)

    def create_src_pad_mask(self, matrix: torch.tensor) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src_pad_mask = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                a = matrix[i][j]
                src_pad_mask.append(torch.equal(a, torch.zeros((2, 12, 2)).to(device)))
        src_pad_mask = torch.tensor(src_pad_mask).unsqueeze(0).reshape(matrix.shape[0], -1)
        #         print(src_pad_mask.shape)
        return src_pad_mask

    def encode(self, src):
        return self.transformer.encoder(self.positional_encoder(
            self.xy_embedding(src)))

    def decode(self, tgt, memory, tgt_mask):
        print(self.tgt_tok_emb(tgt).shape)
        return self.transformer.decoder(self.positional_encoder(
            self.tgt_tok_emb(tgt).squeeze(0)), memory,
            tgt_mask)


def build_model(path, device):
    model = Optimus_Prime(
        num_tokens=5, dim_model=1024, num_heads=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout_p=0
    ).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()


def predict(model, input_sequence, SOS_token=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    length = len(input_sequence[0])

    for _ in range(length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, tgt_mask)
        next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)
        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

    result = y_input.view(-1).tolist()[1:]               # list

    return result

def get_mp(mp_dict, name):
    for mp in mp_dict['points']:
        if mp['name'] == name:
            return mp['ck']
    return "mp not found"


def get_court_length(mp):
    a = mp[4][1] - mp[0][1]
    b = mp[5][1] - mp[1][1]
    return (a+b)/2


def get_court_width(mp):
    a = mp[1][0] - mp[0][0]
    b = mp[5][0] - mp[4][0]
    return (a+b)/2


def top_bottom(joint):
    a = joint[0][15][1] + joint[0][16][1]
    b = joint[1][15][1] + joint[1][16][1]
    if a > b:
        top = 1
        bottom = 0
    else:
        top = 0
        bottom = 1
    return top, bottom

def get_data(path, keypoints):
    x_avg = 0.8931713777441783
    x_stdev = 0.16368038263140244
    y_avg = 0.8750905019870867
    y_stdev = 0.2515293240145384

    norm_num_x = get_court_width(keypoints)
    norm_num_y = get_court_length(keypoints)

    joint_data = []

    with open(path, 'r') as score_json:
        frame_dict = json.load(score_json)

        for i in range(len(frame_dict['frames'])):
            temp_x = []
            joint = frame_dict['frames'][i]['joint']

            for player in range(2):
                for jp in range(17):
                    joint[player][jp][0] = joint[player][jp][0] / norm_num_x
                    joint[player][jp][1] = joint[player][jp][1] / norm_num_y
                    joint[player][jp][0] = (joint[player][jp][0] - x_avg) / x_stdev
                    joint[player][jp][1] = (joint[player][jp][1] - y_avg) / y_stdev

            top, bot = top_bottom(joint)

            if top != 1:
                t = []
                t.append(joint[bot])
                t.append(joint[top])
                joint = np.array(t)

            for p in range(2):
                temp_x.append(joint[p][5:])
            joint_data.append(temp_x)

    return joint_data
