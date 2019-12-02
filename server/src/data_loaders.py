import base64

from torchvision import datasets
from torchvision import transforms
from PIL import Image
from io import BytesIO
from os import path


transforms_data_without_normalize = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


transform_data = transforms.Compose(
    [
        transforms_data_without_normalize,
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def load_dataset(image_path: str):
    data = datasets.ImageFolder(root=image_path, transform=transform_data)
    return data


def load_base64_image(base64_data: str, do_normalize=False):
    image = Image.open(BytesIO(base64.b64decode(base64_data)))
    if do_normalize:
        image = transform_data(image)
    else:
        image = transforms_data_without_normalize(image)
    image = image.float().unsqueeze(0)
    return image


def load_classes(filename: str):
    class_names = [
        line.strip().split(", ", 1) for line in open(path.join("./data", filename))
    ]
    return [line[1] for line in class_names]
