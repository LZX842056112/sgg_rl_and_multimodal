import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from data.data_utils import tokenizer, get_mean_std

class FMNIST(Dataset):
    def __init__(self,
                 config,
                 train=True,
                 download=True,
                 transform=None
            ):

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(config.img_size),
                T.ToTensor()
            ])


        self.dataset = FashionMNIST(root=config.data_location, train=train, download=download, transform=T.Resize(config.img_size))

        self.text_seq_length = config.text_seq_length
        # 由于数据集中只有图片和分类标签，所以我们需要合成一些图片对应的文本
        self.captions = {
            0: "An image of a t-shirt/top",
            1: "An image of trousers",
            2: "An image of a pullover",
            3: "An image of a dress",
            4: "An image of a coat",
            5: "An image of a sandal",
            6: "An image of a shirt",
            7: "An image of a sneaker",
            8: "An image of a bag",
            9: "An image of an ankle boot"
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image = self.transform(self.dataset[i][0])

        caption, mask = tokenizer(self.captions[self.dataset[i][1]], text_seq_length=self.text_seq_length)

        return {"image": image, "caption": caption, "mask": mask}


class DatasetSplit(Dataset):
    def __init__(self,
                 data,
                 captions=None,
                 transform=T.Compose([])
            ):
        self.dataset = data

        self.captions = captions

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image = self.transform(self.dataset[i]["image"])
        caption = self.dataset[i]["caption"]
        mask = self.dataset[i]["mask"]

        return {"image": image, "caption": caption, "mask": mask}

def get_train_set(config, augment_data=False):
    """获取训练数据集"""
    if config.dataset == "fashion_mnist":
        dataset = FMNIST(config, train=True, download=True, transform=T.Resize(config.img_size))
    else:
        raise Exception("Dataset not implemented.")

    mean, std = get_mean_std(dataset, config.img_channels, denom=255)

    transform = [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]
    # 数据增强部分
    if augment_data and config.prob_hflip > 0:
        transform = [T.RandomHorizontalFlip(p=config.prob_hflip)] + transform

    if augment_data and config.crop_padding != 0:
        transform = [T.RandomCrop(config.img_size[0], padding=config.crop_padding)] + transform

    transform = T.Compose([T.Resize(config.img_size)] + transform)

    dataset.transform = transform

    return dataset, mean, std

def get_train_val_split(config, augment_data=False):
    if config.dataset == "fashion_mnist":
        dataset = FMNIST(config, train=True, download=True, transform=T.Resize(config.img_size))
    else:
        raise Exception("Dataset does not exist")

    train_set, val_set = torch.utils.data.random_split(dataset, config.train_val_split)
    mean, std = get_mean_std(train_set, config.img_channels, denom=255)

    transform = [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]

    val_transform = T.Compose(transform)

    val_set = DatasetSplit(val_set, captions=dataset.captions, transform=val_transform)

    if augment_data and config.prob_hflip > 0:
        transform = [T.RandomHorizontalFlip(p=config.prob_hflip)] + transform

    if augment_data and config.crop_padding != 0:
        transform = [T.RandomCrop(config.img_size[0], padding=config.crop_padding)] + transform

    train_transform = T.Compose(transform)

    train_set = DatasetSplit(train_set, captions=dataset.captions, transform=train_transform)

    return train_set, val_set, mean, std

def get_test_set(config, mean=None, std=None):
    if config.dataset == "fashion_mnist":
        dataset = FMNIST(config, train=True, download=False, transform=T.Resize(config.img_size))
    else:
        raise Exception("Dataset not implemented.")

    if mean is None or std is None:
        mean, std = get_mean_std(dataset, config.img_channels, denom=255)

    transform = T.Compose([
        T.Resize(config.img_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    if config.dataset == "fashion_mnist":
        dataset = FMNIST(config, train=False, download=False, transform=transform)
    else:
        raise Exception("Dataset does not exist")

    return dataset