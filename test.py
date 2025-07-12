# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import random
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import utils.score_calculation as lib
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std, save_as_dataframe
    from utils.data_loader import seed_everything, CIFAR_dataloader, CIFAR_oodloader, IMAGENET_dataloader, IMAGENET_oodloader
    from models.wrn import WideResNet
    from models.densenet import DenseNet3
    from models.resnet import SupCEHeadResNet
    from models.resnet import resnet18, resnet34

parser = argparse.ArgumentParser(description='Evaluates a OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--mode', default='TEST', type=str)
parser.add_argument('--data_dir', type=str, default='/home/jw0112/data', help='Path to data directory.')
parser.add_argument('--dataset', '-m', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'IMAGENET100'],
                    help='Name of ID dataset.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--img_size', '-i', type=int, default=32, help='Image size.')
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--seed', default=502, type=int)
parser.add_argument('--ood_dataset_list',
                    default=['SVHN', 'TEXTURES', 'PLACES365', 'iSUN', 'LSUN-C', 'LSUN-R'],
                    # default=['iNaturalist', 'ImageNet-o', 'SUN', 'PLACES', 'TEXTURES'],
                    type=str, nargs='+', help='List of OOD datasets to evaluate on.')
parser.add_argument('--custom_dataset', default='custom', type=str, help='Custom dataset name for ImageFolder.')

# Loading details
parser.add_argument('--model_name', default='wrn', type=str)
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--feat_dim', default=128, type=int, help='feature dim')
parser.add_argument('--load', '-l', type=str,
                    default='./checkpoint/CIFAR100/EOD_CIFAR100_resnet34_seed502_lr0.1_kl0.5_thres0.9_scale5.0_epoch100.pt',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='ENERGY', type=str, choices=['MSP', 'ENERGY', 'ODIN', 'M'],
                    help='Type of score to use for OOD detection')
parser.add_argument('--T', default=1.0, type=float, help='temperature: ENERGY|ODIN') # ENERGY=1 / ODIN=1000
parser.add_argument('--noise', type=float, default=0., help='noise for ODIN') # CIFAR10 : 0.0014 / CIFAR100 : 0.002

args = parser.parse_args()
print(args)

#%%
def get_ood_scores(args, loader, net, in_dist=False):
    _score, _right_score, _wrong_score = [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                score = to_np((output.mean(1) - torch.logsumexp(output, dim=1)))
            elif args.score == 'ENERGY':
                score = -to_np((args.T * torch.logsumexp(output / args.T, dim=1)))
            else:
                score = -np.max(smax, axis=1)
            _score.append(score)

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def compute_in_scores(args, test_loader, net, ood_num_examples, num_classes):
    """Computes in-distribution scores and required parameters for OOD scoring.
    Returns:
        in_score: ndarray of in-distribution scores
        right_score: scores of correctly predicted samples
        wrong_score: scores of incorrectly predicted samples
        extra_params: dict containing Mahalanobis-related data if needed
    """
    if args.score == 'ODIN':
        in_score, right_score, wrong_score = lib.get_ood_scores_odin(
            test_loader, net, args.test_bs, ood_num_examples,
            args.T, args.noise, in_dist=True
        )
        return in_score, right_score, wrong_score, {}

    elif args.score == 'M':
        from torch.autograd import Variable
        _, right_score, wrong_score = get_ood_scores(args, test_loader, net, in_dist=True)

        num_batches = ood_num_examples // args.test_bs

        temp_x = Variable(torch.rand(2,3,32,32)).cuda()
        temp_list = net.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)

        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        print('get sample mean and covariance', count)

        sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
        in_score = lib.get_Mahalanobis_score(
            net, test_loader, num_classes, sample_mean, precision, count-1,
            args.noise, num_batches, in_dist=True
        )
        print(in_score[-3:], in_score[-103:-100])

        extra_params = {
            'sample_mean': sample_mean,
            'precision': precision,
            'layer_index': count - 1,
            'num_batches': num_batches,
            'num_classes': num_classes,
        }
        return in_score, right_score, wrong_score, extra_params

    else:
        in_score, right_score, wrong_score = get_ood_scores(args, test_loader, net, in_dist=True)
        return in_score, right_score, wrong_score, {}
    
def get_and_print_results(args, ood_loader, net, in_score, extra_params):
    """Evaluates and prints AUROC, AUPR, FPR scores for given OOD loader.
    """
    aurocs, auprs, fprs = [], [], []
    
    for _ in range(args.num_to_avg):
        if args.score == 'ODIN':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, len(in_score), args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(
                net, ood_loader, extra_params['num_classes'],
                extra_params['sample_mean'], extra_params['precision'],
                extra_params['layer_index'], args.noise,
                extra_params['num_batches']
            )
        else:
            out_score = get_ood_scores(args, ood_loader, net)

        if args.out_as_pos:
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)

    if args.num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.dataset)
    else:
        print_measures(auroc, aupr, fpr, args.dataset)

    return auroc, aupr, fpr

#%%
def main(args):
    in_score, right_score, wrong_score, extra_params = compute_in_scores(args, test_loader, net, ood_num_examples, num_classes)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

    args.log_directory = f'./results/{args.dataset}/{args.score}'
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    # /////////////// Error Detection ///////////////

    print('\n\nError Detection')
    show_performance(wrong_score, right_score, method_name=args.dataset)

    auroc_list, aupr_list, fpr_list = [], [], []

    for name, ood_loader in ood_loaders.items():
        print(f"\n{name} Detection")
        auroc, aupr, fpr = get_and_print_results(args, ood_loader, net, in_score, extra_params)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    print('\n\nMean Test Results!!!!!')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.dataset)
    save_as_dataframe(args, args.ood_dataset_list, fpr_list, auroc_list, aupr_list)

if __name__ == '__main__':
    # cudnn.benchmark = True  # fire on all cylinders
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'CIFAR' in args.dataset:
        train_dataset, test_dataset, train_loader, test_loader, num_classes = CIFAR_dataloader(args)
        ood_loaders = CIFAR_oodloader(args)
    elif 'IMAGENET100' == args.dataset:
        train_dataset, test_dataset, train_loader, test_loader, num_classes = IMAGENET_dataloader(args)
        ood_loaders = IMAGENET_oodloader(args)

    # Create model
    if args.model_name == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    elif args.model_name == 'densenet':
        net = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True,
                        dropRate=0.0, normalizer=None, k=None, info=None)
    elif args.model_name == 'resnet18':
        # net = SupCEHeadResNet(name=args.model_name, feat_dim=args.feat_dim, num_classes=num_classes)
        net = resnet18(num_classes=num_classes)
    elif args.model_name == 'resnet34':
        net = resnet34(num_classes=num_classes)
    else:
        raise ValueError('Unknown model name: {}'.format(args.model_name))

    state_dict = torch.load(args.load, map_location='cpu')
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    net.load_state_dict(state_dict)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu))).to(device) if args.ngpu > 1 else net.to(device)
    net.eval()

    ood_num_examples = len(test_dataset) // 5
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    
    main(args)