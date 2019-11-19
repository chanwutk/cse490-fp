import torch.nn as nn
import torch.nn.functional as F
import pt_util
import torchvision.models as models


class VisualizableSequential(nn.Sequential):
    def __init__(self, visualizable=False, *args):
        super(VisualizableSequential, self).__init__(*args)
        self.visualizable = visualizable
        if visualizable:
            self.visualizations = []
        else:
            self.visualizations = None

    def forward(self, input):
        for module in self._modules.values():
            if self.visualizable:
                module_input = input.numpy()
            input = module(input)
            if self.visualizable:
                module_output = input.numpy()
                weight = None
                if hasattr(module, "weight"):
                    weight = module.weight.numpy()
                self.visualizations.append(
                    (module_input, module_output, str(module), weight)
                )
        return input

    def get_visualizations(self):
        return self.visualizations


class VisualizableVgg(nn.Module):
    def __init__(self, save=True, visualizable=False, batch_norm=True, *args, **kwargs):
        self.visualizable = visualizable
        self.save = save
        self.__best_accuracy
        self.vgg = models.vgg19_bn()
        self.vgg.features = VisualizableSequential(self.vgg.features.layer)

    def get_visualizations(self):
        return self.vgg.features.get_visualizations()

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
