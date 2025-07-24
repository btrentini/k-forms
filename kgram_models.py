import torch
import torch.nn as nn
from math import comb


class Kgrams(nn.Module):
    def __init__(self, l, n, k, hidden_dim, n_classes):
        super().__init__()

        self.in_dim = n
        self.combination = comb(n, k)
        self.l = l
        self.out_dim = self.combination * l
        self.classhead_input_size = l * l

        self.diffnet = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
            nn.Unflatten(2, torch.Size([self.combination, l]))
        )
        self.classhead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.classhead_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, PG, P, mask=None):
        if P.shape[1] == 0:
            return torch.zeros((P.shape[0], self.l, self.l))
        w = self.diffnet(P)
        # TODO: DRAGONS LIVE HERE. Review.
        kgram = torch.einsum("...pcl, ...pcb, ...pbj -> ...plj", w, PG, w)
        if mask is not None:
            if torch.all(mask == False):
                kgram = torch.zeros_like(kgram)
            else:
                kgram = kgram[..., mask[-1], :, :]

        # dim = kgram.ndim - 4
        dim = -3  # Fix this
        # print(f"kgram shape: {kgram.shape}, mask_nontrivial: {torch.sum(mask) if mask is not None else None}")
        pcreps = torch.sum(kgram, dim=dim)  # sums only for PC classification
        # p_classes = self.classhead(pcreps)++
        dia = torch.diagonal(pcreps, dim1=-1, dim2=-2)
        return dia

    def evaluate_kforms(self, x):
        return self.diffnet(torch.unsqueeze(x, 0))
