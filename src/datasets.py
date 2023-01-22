import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# the testing transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def get_data_loader(args, use_cuda):
    train_dataset = datasets.ImageFolder(
        root=f'{args.data_dir}/train',
        transform=train_transform
    )
    # validation dataset
    valid_dataset = datasets.ImageFolder(
        root=f'{args.data_dir}/valid',
        transform=valid_transform
    )
    # testing dataset
    test_dataset = datasets.ImageFolder(
        root=f'{args.data_dir}/test',
        transform=test_transform
    )

    # kwargs
    train_kwargs = {'batch_size': args.batch_size}
    valid_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        valid_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # data loaders
    train_loader = DataLoader(train_dataset, **train_kwargs)
    valid_loader = DataLoader(valid_dataset, **valid_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, valid_loader, test_loader