import torch
import base64
import io

from torchvision.transforms import functional as TF


def get_first_or_value_of(x):
    if type(x).__name__ == "int":
        return x
    elif type(x).__name__ == "tuple":
        return x[0]
    return None


def to_layer_info(trace):
    _, _, layer, _ = trace
    layer_type = type(layer).__name__
    layer_info = {"type": layer_type, "str": str(layer)}
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
    min_value = torch.min(tensor).item()
    max_value = torch.max(tensor).item()
    range_value = max_value - min_value
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
    tensor = tensor.squeeze(dim=0)
    for i in range(tensor.size()[0]):
        normalized = normalize_tensor(tensor[i, :, :].squeeze(dim=0))
        image = TF.to_pil_image(normalized)
        base64s.append(pil_to_base64(image))
    return base64s


def weights_to_base64s(weights):
    base64s = [None] * weights.size()[0]
    for i in range(weights.size()[0]):
        base64s[i] = [None] * weights.size()[1]
    for wo in range(weights.size()[0]):
        out_kernels = weights[wo, :, :, :].squeeze(dim=0)
        normalized = normalize_tensor(out_kernels)
        for wi in range(weights.size()[1]):
            kernel_normalized = normalized[wi, :, :].squeeze(dim=0)
            image = TF.to_pil_image(kernel_normalized)
            base64s[wo][wi] = {
                "data": pil_to_base64(image),
            }
    return base64s


def weight_to_base64(weights, wi: int, wo: int):
    out_kernels = weights[wo, :, :, :].squeeze(dim=0)
    normalized = normalize_tensor(out_kernels)

    kernel_normalized = normalized[wi, :, :].squeeze(dim=0)
    image = TF.to_pil_image(kernel_normalized)
    return {
        "data": pil_to_base64(image),
    }
