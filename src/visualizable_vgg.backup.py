import torch
import torch.nn as nn
import torch.nn.functional as F
import pt_util


class VisualizableVgg(nn.Module):
    def __init__(self, save=True, vis=False):
        super(VisualizableVgg, self).__init__()
        self.__vis = vis
        self.__best_accuracy
        # 224 * 224 * 3
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # 224 * 224 * 64
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 224 * 224 * 64
        # maxpool - 2 * 2
        # 112 * 112 * 64
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 112 * 112 * 128
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # 112 * 112 * 128
        # maxpool - 2 * 2
        # 56 * 56 * 128
        self.c5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # 56 * 56 * 256
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 56 * 56 * 256
        self.c7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 56 * 56 * 256
        # maxpool - 2 * 2
        # 28 * 28 * 256
        self.c8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 28 * 28 * 512
        self.c9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 28 * 28 * 512
        self.c10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 28 * 28 * 512
        # maxpool - 2 * 2
        # 14 * 14 * 512
        self.c11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 14 * 14 * 512
        self.c12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 14 * 14 * 512
        self.c13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 14 * 14 * 512
        # maxpool - 2 * 2
        # 7 * 7 * 512
        self.f1 = nn.Linear(7 * 7 * 512, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 5)
        self.act = nn.LeakyReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(x1))
        x2 = self.mp(x2)

        x3 = self.act(self.c3(x2))
        x4 = self.act(self.c4(x3))
        x4 = self.mp(x4)

        x5 = self.act(self.c3(x4))
        x6 = self.act(self.c4(x5))
        x7 = self.act(self.c4(x6))
        x7 = self.mp(x7)

        x8 = self.act(self.c3(x7))
        x9 = self.act(self.c4(x8))
        x10 = self.act(self.c4(x9))
        x10 = self.mp(x10)

        x11 = self.act(self.c3(x10))
        x12 = self.act(self.c4(x11))
        x13 = self.act(self.c4(x12))
        x14 = self.mp(x13)

        x_flat = torch.flatten(x14, 1)

        x15 = self.act(self.f1(x_flat))
        x15 = F.dropout(x6)

        x16 = self.act(self.f2(x15))

        return F.log_softmax(self.f3(x16), dim=1)

    def visualizable_layer(self, act, layer, x):
        if self.__vis:
            layer_input = x.clone()
        layer_output = act(layer(x))
        w = layer.grad.clone()
        return layer_input, layer_output, w

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
