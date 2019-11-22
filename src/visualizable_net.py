import torch.nn as nn
import torch.nn.functional as F
import src.pt_util as pt_util
import torchvision.models as models


class VisualizableSequential(nn.Sequential):
    def __init__(self, args, visualizable=False):
        super(VisualizableSequential, self).__init__(*args)
        self.visualizable = visualizable
        self.visualizations = []

    def forward(self, input):
        if self.visualizable:
            self.visualizations = []
        for module in self._modules.values():
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
        return input

    def get_visualizations(self):
        return self.visualizations

    def set_visualizable(self, visualizable=True):
        self.visualizable = visualizable

    def __str__(self):
        return super(VisualizableSequential, self).__str__()


class BaseSaveableNet(nn.Module):
    def __init__(self):
        super(BaseSaveableNet, self).__init__()
        self.__best_accuracy_saved = None

    def classify(self, input):
        return F.softmax(self.forward(input), dim=1)

    def loss(self, prediction, label, reduction="mean"):
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


def make_visualizable_net(net_builder, num_classes, visualizable):
    net = net_builder(num_classes=num_classes)
    net.features = VisualizableSequential(list(net.features.modules())[1:])
    net.features.set_visualizable(visualizable)
    return net


class VisualizableAlexNet(BaseSaveableNet):
    def __init__(self, num_classes, visualizable=False):
        super(VisualizableAlexNet, self).__init__()
        self.visualizable = visualizable
        self.alexnet = make_visualizable_net(
            models.alexnet, num_classes=num_classes, visualizable=visualizable
        )

    def forward(self, input):
        return self.alexnet(input)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)

    def set_visualizable(self, visualisable=True):
        self.visualizable = visualisable
        self.alexnet.features.set_visualization(visualisable)

    def get_visualizations(self):
        return self.alexnet.features.get_visualizations()


class VisualizableVgg(BaseSaveableNet):
    def __init__(self, num_classes, visualizable=False):
        super(VisualizableVgg, self).__init__()
        self.visualizable = visualizable
        self.vgg = make_visualizable_net(
            models.vgg19_bn, num_classes=num_classes, visualizable=visualizable
        )

    def forward(self, input):
        return self.vgg(input)

    def set_visualizable(self, visualisable=True):
        if visualisable != self.visualizable:
            self.visualizable = visualisable
            self.vgg.features.set_visualization(visualisable)

    def get_visualizations(self):
        return self.vgg.features.get_visualizations()
