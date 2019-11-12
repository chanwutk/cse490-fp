import torch
import torch.nn as nn
import torch.nn.functional as F
import pt_util


class FlowerNetwork(nn.Module):
    def __init__(self, save=True, vis=False):
        super(FlowerNetwork, self).__init__()
        self.__vis = vis
        self.__best_accuracy
        # 224 * 224 * 3
        self.c1 = nn.Conv2d(3, 96, kernel_size=11, padding=5, stride=4)
        # 56 * 56 * 96
        self.c2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        # maxpool - 2 * 2
        # 28 * 28 * 256
        self.c3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        # maxpool - 2 * 2
        # 14 * 14 * 384
        self.c4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        # 14 * 14 * 384
        self.c5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # 14 * 14 * 256
        # maxpool - 2 * 2
        # 7 * 7 * 256
        self.f1 = nn.Linear(7 * 7 * 256, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 5)
        self.act = nn.LeakyReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.act(self.c1(x))

        x2 = self.act(self.c2(x1))
        x2 = self.mp(x2)

        x3 = self.act(self.c3(x2))
        x3 = self.mp(x3)

        x4 = self.act(self.c4(x3))

        # adding residual
        x5 = self.act(self.c5(x4 + x3))
        x5 = self.mp(x5)

        x_flat = torch.flatten(x5, 1)

        x6 = self.act(self.f1(x_flat))
        x6 = F.dropout(x6)

        x7 = self.act(self.f2(x6))

        return F.log_softmax(self.f3(x7), dim=1)

    def loss(self, prediction, label, reduction="elementwise_mean"):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if self.__best_accuracy is None or self.__best_accuracy < accuracy:
            self.__best_accuracy = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)
