import base64

from torchvision import datasets
from torchvision import transforms
from PIL import Image
from io import BytesIO


transform_data = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def load_dataset(image_path: str):
    data = datasets.ImageFolder(root=image_path, transform=transform_data)
    return data


def load_base64_image(base64_data: str):
    image = Image.open(BytesIO(base64.b64decode(base64_data)))
    image = transform_data(image).float()
    image = image.unsqueeze(0)
    return image


def load_classes():
    class_names = [line.strip().split(", ") for line in open("./data/class_names.txt")]
    return [line[1] for line in class_names]
