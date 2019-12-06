import torch
import torch.nn as nn


class NeuralLearner(nn.Module):
    def __init__(self, opt=(3, 10, 10, 1)):
        super(NeuralLearner, self).__init__()

        self.layers = list()

        for i in range(len(opt) - 1):
            self.layers.append(
                nn.Linear(
                    in_features=opt[i],
                    out_features=opt[i + 1]
                )
            )

            if i < len(opt) - 1:
                self.layers.append(
                    nn.LeakyReLU()
                )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x

        for layer in self.layers:
            out = layer(out)

        return out
