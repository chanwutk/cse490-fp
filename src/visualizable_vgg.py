import torch.nn as nn
import torch.nn.functional as F
import pt_util
import torchvision.models as models


class VisualizableVgg(nn.Module):
    def __init__(self, save=True, vis=False):
        super(VisualizableVgg, self).__init__()
        self.__vis = vis
        self.__best_accuracy
        self.save = save
        self.vgg = models.vgg16_bn()

    def forward(self, x):
        return self.vgg(x)

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
