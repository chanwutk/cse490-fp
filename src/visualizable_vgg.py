import torch
import torch.nn as nn
import torch.nn.functional as F
import pt_util
import torchvision.models as models


class VisualizableSequential(nn.Sequential):
    def __init__(self, args, visualizable=False):
        super(VisualizableSequential, self).__init__(*args)
        self.visualizable = visualizable
        self.visualizations = []

    def forward(self, input):
        for module in self._modules.values():
            # print(module)
            # print(input.size())
            if self.visualizable:
                module_input = input.detach()
            input = module(input)
            if self.visualizable:
                module_output = input.detach()
                weight = None
                if hasattr(module, "weight"):
                    weight = module.weight.detach()
                self.visualizations.append(
                    (module_input, module_output, str(module), weight)
                )
        # print('done sequential')
        return input

    def get_visualizations(self):
        return self.visualizations

    def set_visualizable(self, visualizable=True):
        if visualizable != self.visualizations:
            self.visualizable = visualizable

    def __str__(self):
        return super(VisualizableSequential, self).__str__()


class VisualizableVgg(nn.Module):
    def __init__(self, save=True, visualizable=False, batch_norm=True):
        super(VisualizableVgg, self).__init__()
        self.visualizable = visualizable
        self.save = save
        self.__best_accuracy_saved = None
        if batch_norm:
            self.vgg = models.vgg19_bn(num_classes=5)
        else:
            self.vgg = models.vgg19(num_classes=5)
        # print(self.vgg.features)
        self.vgg.features = VisualizableSequential(
            list(self.vgg.features.modules())[1:]
        )
        self.vgg.features.set_visualizable(False)
        # print(self.vgg.features)

    def forward(self, input):
        return self.vgg(input)

    def loss(self, prediction, label, reduction="elementwise_mean"):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if self.__best_accuracy_saved is None or self.__best_accuracy_saved < accuracy:
            self.__best_accuracy_saved = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)

    def set_visualizable(self, visualisable=True):
        if visualisable != self.visualizable:
            self.visualizable = visualisable
            self.vgg.features.set_visualization(visualisable)

    def get_visualizations(self):
        return self.vgg.features.get_visualizations()
