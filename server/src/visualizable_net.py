import torch.nn as nn
import torch.nn.functional as F
import src.pt_util as pt_util
import torchvision.models as models


class TraceableSequential(nn.Sequential):
    def __init__(self, args, traceable=False):
        super(TraceableSequential, self).__init__(*args)
        self.traceable = traceable
        self.traces = []

    def forward(self, input):
        if self.traceable:
            self.traces = []
        for module in self._modules.values():
            if self.traceable:
                module_input = input.detach()
            input = module(input)
            if self.traceable:
                module_output = input.detach()
                weight = None
                if hasattr(module, "weight"):
                    weight = module.weight.detach()
                self.traces.append((module_input, module_output, module, weight))
        return input

    def get_traces(self):
        return self.traces

    def set_traceable(self, traceable=True):
        self.traceable = traceable

    def __str__(self):
        return super(TraceableSequential, self).__str__()


class BaseSavableNet(nn.Module):
    def __init__(self):
        super(BaseSavableNet, self).__init__()
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


def make_traceable_net(net_builder, num_classes, traceable):
    net = net_builder(num_classes=num_classes)
    net.features = TraceableSequential(
        list(net.features.modules())[1:], traceable=traceable
    )
    return net


class TraceableAlexNet(BaseSavableNet):
    def __init__(self, num_classes, traceable=False):
        super(TraceableAlexNet, self).__init__()
        self.traceable = traceable
        self.alexnet = make_traceable_net(
            models.alexnet, num_classes=num_classes, traceable=traceable
        )

    def forward(self, input):
        return self.alexnet(input)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)

    def set_traceable(self, traceable=True):
        self.traceable = traceable
        self.alexnet.features.set_traceable(traceable)

    def get_traces(self):
        return self.alexnet.features.get_traces()


class TraceableVgg(BaseSavableNet):
    def __init__(self, num_classes, traceable=False):
        super(TraceableVgg, self).__init__()
        self.traceable = traceable
        self.vgg = make_traceable_net(
            models.vgg19_bn, num_classes=num_classes, traceable=traceable
        )

    def forward(self, input):
        return self.vgg(input)

    def set_traceable(self, traceable=True):
        self.traceable = traceable
        self.vgg.features.set_traceable(traceable)

    def get_traces(self):
        return self.vgg.features.get_traces()
