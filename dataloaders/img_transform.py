from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop, RandomHorizontalFlip
from torchvision import transforms
from PIL import Image


MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def get_transform(image_resolution, is_training, version='v1'):
    if is_training:
        transform = Compose([
                    Resize(image_resolution, interpolation=Image.BICUBIC),
                    RandomResizedCrop(image_resolution, scale=(0.7, 1.0), ),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize(MEAN, STD),
                ])   
    else:
        transform = Compose([
            Resize(image_resolution, interpolation=Image.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(MEAN, STD),
        ])
    return transform
