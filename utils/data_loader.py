import os, random
import math
import numpy as np
from PIL import Image
import torch
import torch.utils
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import Subset, DataLoader
from utils import svhn_loader as svhn
from utils import lsun_loader as lsun_loader

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]
cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

dset_info = {'CIFAR10': (cifar10_mean, cifar10_std, 10),
             'CIFAR100': (cifar100_mean, cifar100_std, 100),
             'IMAGENET100': (imagenet_mean, imagenet_std, 100)}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CIFAR_dataloader(args):
    mean, std, num_classes = dset_info[args.dataset]
    
    train_transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(args.img_size, padding=4),
        trn.Resize((args.img_size, args.img_size)),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std)
    ])
    
    test_transform = trn.Compose([
        trn.Resize((args.img_size, args.img_size)),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std)
    ])
    
    if args.dataset == 'CIFAR10':
        train_dataset = dset.CIFAR10(root=args.data_dir, train=True, download=False,
                                     transform=train_transform if args.mode != 'TEST' else test_transform)
        test_dataset = dset.CIFAR10(root=args.data_dir, train=False, download=False, transform=test_transform)
    
    elif args.dataset == 'CIFAR100':
        train_dataset = dset.CIFAR100(root=args.data_dir, train=True, download=False,
                                      transform=train_transform if args.mode != 'TEST' else test_transform)
        test_dataset = dset.CIFAR100(root=args.data_dir, train=False, download=False, transform=test_transform)
    
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size if args.mode != 'TEST' else args.test_bs,
                                               shuffle=True, num_workers=args.prefetch, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_bs,
                                              shuffle=False, num_workers=args.prefetch, pin_memory=True)
    
    return train_dataset, test_dataset, train_loader, test_loader, num_classes


def CIFAR_oodloader(args):
    mean, std, _ = dset_info[args.dataset]
    # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    ood_dataloaders = {}
    
    if args.mode == 'OE':
        ood_images = np.load(os.path.join(args.data_dir, args.oe_dataset_name))
        ood_images = torch.from_numpy(ood_images).permute(0, 3, 1, 2)
        ood_dataset = torch.utils.data.TensorDataset(ood_images, torch.zeros_like(ood_images, dtype=torch.long))
        ood_loader = DataLoader(ood_dataset, batch_size=args.oe_batch_size, shuffle=False,
                                num_workers=args.prefetch, pin_memory=True, drop_last=True)
        ood_dataloaders[args.mode] = ood_loader
    
    else:
        for name in args.ood_dataset_list:
            if name == 'CIFAR10':
                ood_dataset = dset.CIFAR10(root=args.data_dir, train=False, download=False,
                                        transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
            elif name == 'CIFAR100':
                ood_dataset = dset.CIFAR100(root=args.data_dir, train=False, download=False,
                                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
            elif name == 'SVHN':
                ood_dataset = svhn.SVHN(root=args.data_dir, split='test',
                                        transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
            elif name == 'PLACES365':
                ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'places365_test'), # places365_test / places365_test_subset
                                            transform=trn.Compose([trn.Resize(args.img_size),
                                                                    trn.CenterCrop(args.img_size),
                                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
            elif name == 'TEXTURES':
                ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'dtd/dtd/images'),
                                            transform=trn.Compose([trn.Resize(args.img_size),
                                                                    trn.CenterCrop(args.img_size),
                                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
            elif name == 'LSUN-C':
                ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'LSUN-C'),
                                            transform=trn.Compose([ # trn.Resize(args.img_size), trn.CenterCrop(args.img_size),
                                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
            elif name == 'LSUN-R':
                ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'LSUN-R'),
                                            transform=trn.Compose([ # trn.Resize(args.img_size), trn.CenterCrop(args.img_size),
                                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
            elif name == 'iSUN':
                ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'iSUN'),
                                            transform=trn.Compose([trn.Resize(args.img_size),
                                                                    #   trn.CenterCrop(args.img_size),
                                                                    trn.ToTensor(), trn.Normalize(mean, std)]))        
            else:
                ood_dataset = dset.ImageFolder(f'{args.data_dir}/{args.custom_dataset}',
                                            transform=trn.Compose([trn.Resize(args.img_size),
                                                                    trn.CenterCrop(args.img_size),
                                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
            # if len(ood_dataset) > 10000:
            #     ood_dataset = Subset(ood_dataset, np.random.choice(len(ood_dataset), 10000, replace=False))
            ood_loader = DataLoader(ood_dataset, batch_size=args.test_bs, shuffle=False,
                                                    num_workers=args.prefetch, pin_memory=True)
            ood_dataloaders[name] = ood_loader
    return ood_dataloaders

def IMAGENET_dataloader(args):
    mean, std, num_classes = dset_info[args.dataset]
    
    train_transform = trn.Compose([
        trn.RandomResizedCrop(size=args.img_size, interpolation=trn.InterpolationMode.BICUBIC),
        trn.RandomHorizontalFlip(p=0.5),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std)
    ])
    
    test_transform = trn.Compose([
        trn.Resize(size=(args.img_size, args.img_size), interpolation=trn.InterpolationMode.BICUBIC),
        trn.CenterCrop(size=(args.img_size, args.img_size)),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std)
    ])
    
    if args.mode != 'TEST':
        train_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'imagenet100/train'), transform=train_transform)
    else:
        train_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'imagenet100/train'), transform=test_transform)
    
    valid_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'imagenet100/val'), transform=test_transform)
    test_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'imagenet100/val'), transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size if args.mode != 'TEST' else args.test_bs,
                                               shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_bs, shuffle=False,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)
    
    return train_dataset, test_dataset, train_loader, test_loader, num_classes

def IMAGENET_oodloader(args):
    mean, std, _ = dset_info[args.dataset]
    # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    ood_dataloaders = {}
    
    for name in args.ood_dataset_list:
        if name == 'iNaturalist':
            ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'iNaturalist/test'),
                                        transform=trn.Compose([trn.Resize(args.img_size),
                                                                trn.CenterCrop(args.img_size),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))
        elif name == 'ImageNet-o':
            ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'imagenet-o'),
                                        transform=trn.Compose([trn.Resize(args.img_size),
                                                                trn.CenterCrop(args.img_size),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))
        elif name == 'SUN':
            ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'SUN/test'),
                                        transform=trn.Compose([trn.Resize(args.img_size),
                                                                trn.CenterCrop(args.img_size),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))
        elif name == 'PLACES':
            ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'Places'), # places365_test / places365_test_subset
                                        transform=trn.Compose([trn.Resize(args.img_size),
                                                                trn.CenterCrop(args.img_size),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))
        elif name == 'TEXTURES':
            ood_dataset = dset.ImageFolder(root=os.path.join(args.data_dir, 'dtd/dtd/images'),
                                        transform=trn.Compose([trn.Resize(args.img_size),
                                                                trn.CenterCrop(args.img_size),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))      
        else:
            ood_dataset = dset.ImageFolder(f'{args.data_dir}/{args.custom_dataset}',
                                        transform=trn.Compose([trn.Resize(args.img_size),
                                                                trn.CenterCrop(args.img_size),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))
        # if len(ood_dataset) > 10000:
        #     ood_dataset = Subset(ood_dataset, np.random.choice(len(ood_dataset), 10000, replace=False))
        ood_loader = DataLoader(ood_dataset, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)
        ood_dataloaders[name] = ood_loader
    return ood_dataloaders
    
#%%
########################## NPOS ##########################

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_train_transforms(mean, std):  
    train_transform = trn.Compose([
        trn.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        trn.RandomHorizontalFlip(),
        trn.RandomApply([trn.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        trn.RandomGrayscale(p=0.2),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std),
    ])
    return TwoCropTransform(train_transform)

def get_test_transforms(mean, std):
    test_transform = trn.Compose([trn.ToTensor(),
                                  trn.Normalize(mean=mean, std=std),
                                # trn.InterpolationMode.BILINEAR
                                  ])
    return test_transform

def x_u_split(args, labels, expand_labels=True):
    label_ratio = args.label_ratio
    num_labeled = int(label_ratio*len(labels))
    num_unlabeled = len(labels)-num_labeled
    print("Distribution:")
    print(num_labeled, num_unlabeled, len(labels))

    label_per_class = num_labeled // args.n_cls
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.n_cls):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        l_idx = idx[:label_per_class]
        u_idx = idx[label_per_class:]
        labeled_idx.extend(l_idx)
        unlabeled_idx.extend(u_idx)
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == num_labeled
    assert len(unlabeled_idx) == num_unlabeled
    # # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    # unlabeled_idx = np.array(range(len(labels)))

    if expand_labels or num_labeled < args.batch_size:
        num_iter = int(len(unlabeled_idx)/(args.mu*args.batch_size))
        num_expand_x = math.ceil(args.batch_size * num_iter / num_labeled)
        print("Expand:", num_expand_x)
        if num_expand_x != 0:
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class CIFAR10SSL(dset.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None and len(indexs) > 0:
            self.shrink_data(indexs)
            print(len(self.data), len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = torch.from_numpy(targets[idxs])
        self.data = self.data[idxs, ...]
