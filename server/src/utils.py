import torch
import base64
import io

from torchvision.transforms import functional as TF

def get_first_or_value_of(x):
    if type(x).__name__ == 'int':
        return x
    elif type(x).__name__ == 'tuple':
        return x[0]
    return None

def to_layer_info(trace):
    _, _, layer, _ = trace
    layer_type = type(layer).__name__
    layer_info = {"type": layer_type}
    if layer_type == "Conv2d":
        layer_info["inputDim"] = layer.in_channels
        layer_info["outputDim"] = layer.out_channels
        layer_info["stride"] = get_first_or_value_of(layer.stride)
        layer_info["kernelSize"] = get_first_or_value_of(layer.kernel_size)
    elif layer_type == "MaxPool2d":
        layer_info["stride"] = get_first_or_value_of(layer.stride)
        layer_info["kernelSize"] = get_first_or_value_of(layer.kernel_size)
    return layer_info


def normalize_tensor(tensor):
    tensor = tensor.float()
    min_value = torch.min(tensor)
    range_value = torch.max(tensor) - min_value
    if range_value > 0:
        return (tensor - min_value) / range_value
    else:
        return torch.zeros(tensor.size())


def pil_to_base64(pil_image):
    byte_buffer = io.BytesIO()
    pil_image.save(byte_buffer, format="JPEG")

    # reset file pointer to start
    byte_buffer.seek(0)
    img_bytes = byte_buffer.read()

    return base64.b64encode(img_bytes).decode("ascii")


def tensor_to_base64s(tensor):
    base64s = []
    tensor = normalize_tensor(tensor.squeeze(dim=0))
    for i in range(tensor.size()[0]):
        image = TF.to_pil_image(tensor[i, :, :].squeeze(dim=0))
        base64s.append(pil_to_base64(image))
    return base64s


def weights_to_base64s(weights):
    base64s = []
    weights = normalize_tensor(weights)
    for wi in range(weights.size()[0]):
        weight_base64_out = []
        for wo in range(weights.size()[1]):
            image = TF.to_pil_image(weights[wi, wo, :, :].squeeze(dim=0).squeeze(dim=0))
            weight_base64_out.append(pil_to_base64(image))
        base64s.append(weight_base64_out)
    return base64s
