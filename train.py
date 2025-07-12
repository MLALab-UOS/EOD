# -*- coding: utf-8 -*-
import numpy as np
import os, gc
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_grad_cam import GradCAM

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.data_loader import seed_everything, CIFAR_dataloader, CIFAR_oodloader, IMAGENET_dataloader
    from models.allconv import AllConvNet
    from models.wrn import WideResNet

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='EOD', type=str, choices=['CLF', 'VOS', 'NPOS', 'EOD'])
parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'IMAGENET100'],
                    default='CIFAR10', help='Choose between CIFAR10, CIFAR100, IMAGENET100.')
parser.add_argument('--data_dir', type=str, default='/home/jw0112/data', help='Path to dataset.')
parser.add_argument('--model', '-m', type=str, default='wrn', choices=['allconv', 'wrn'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
parser.add_argument('--seed', default=604, type=int)
parser.add_argument('--img_size', '-i', type=int, default=32, help='Image size.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen_factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--penultimate_dim', type=int, default=128, help='penultimate dimesion of the network')

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./checkpoint', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# Specific
parser.add_argument('--start_epoch', type=int, default=80)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--lr_loss_weight', type=float, default=0.1)
parser.add_argument('--kl_loss_weight', type=float, default=0.5)

# EOD Specific
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--scale', type=float, default=5)

# Visualization
parser.add_argument('--ood_dataset_list', default=['SVHN', 'PLACES365'],
                        type=str, nargs='+', help='List of OOD datasets to evaluate on.')
parser.add_argument('--method_name', type=str, default='LR0.1_KL0.5', help='Method name for visualization')
parser.add_argument('--sample_size', type=int, default=3000, help='Number of samples per feature type for visualization')
parser.add_argument('--vis', type=str, default='t-SNE', choices=['t-SNE', 'UMAP'], help="Visualization method")
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

class CheckpointSaver:
    def __init__(self, save_path, mode='max'):
        self.save_path = save_path
        self.mode = mode
        self.best = None

    def save(self, model, metric):
        if self.best is None or \
           (self.mode == 'max' and metric > self.best) or \
           (self.mode == 'min' and metric < self.best):
            print(f'Best model saved with metric: {metric:.2f} at {self.save_path}')
            torch.save(model.state_dict(), self.save_path)
            self.best = metric

class EnergyScoreTarget:
    def __init__(self, category, logistic_regression):
        self.category = category
        self.logistic_regression = logistic_regression

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        energy = -log_sum_exp(model_output, dim=1)
        energy_lr = self.logistic_regression(energy.view(-1, 1)).squeeze()
        return energy_lr[self.category]

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        return m + torch.log(torch.sum(torch.exp(value - m)))

def generate_fgsm_ood(args, images, labels):
    images, labels = images.to(device), labels.to(device)
    correct_mask = (net(images).argmax(dim=1) == labels)
    correct_images = images[correct_mask].detach().clone().requires_grad_(True)
    correct_labels = labels[correct_mask]
    id_labels = torch.ones(len(correct_images), dtype=torch.long).to(device)
    
    with GradCAM(model=net, target_layers=[net.block3.layer[-1].conv2]) as cam:
        targets = [EnergyScoreTarget(l.item(), logistic_regression) for l in id_labels]
        io_cams = torch.tensor(cam(input_tensor=correct_images, targets=targets)).to(device)
    
    epsilon_map = io_cams.div(args.scale).unsqueeze(1).to(device)
    important_region_mask = (io_cams >= args.threshold).unsqueeze(1)
    
    del cam ; torch.cuda.empty_cache() ; gc.collect()

    prediction = net(correct_images)
    
    energy = -log_sum_exp(prediction, dim=1)
    io_prediction = logistic_regression(energy.unsqueeze(1))
    fgsm_loss = F.cross_entropy(io_prediction, id_labels)

    optimizer.zero_grad()
    fgsm_loss.backward()

    img_grads = correct_images.grad.data    
    ascent_perturbation = epsilon_map * img_grads.sign() * important_region_mask.float()
    descent_perturbation = -epsilon_map * img_grads.sign() * (~important_region_mask).float()
    
    semantic_region, nuisance_region = correct_images * important_region_mask, correct_images * (~important_region_mask)
    
    semantic_perturbed = semantic_region + ascent_perturbation
    nuisance_perturbed = nuisance_region + descent_perturbation
    perturbed_image = semantic_perturbed + nuisance_perturbed
    
    fgsm_ood = semantic_perturbed + nuisance_perturbed
    fgsm_ood = torch.clamp(perturbed_image, 0, 1)
    fgsm_ood = fgsm_ood[torch.randperm(len(fgsm_ood))[:args.select * num_classes]]
    
    return fgsm_ood

def train_EOD(epoch):
    global energy_threshold
    net.train()
    loss_avg = 0.0
    energy_score_list = torch.tensor([]).to(device)
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')):
        data, target = data.to(device), target.to(device)
        
        if isinstance(net, torch.nn.DataParallel):
            logit, output = net.module.forward_virtual(data)
        else:
            logit, output = net.forward_virtual(data)

        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]

        lr_reg_loss = torch.zeros(1).to(device)[0]
        kl_loss = torch.zeros(1).to(device)[0]
        
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0)
            
            if epoch == args.start_epoch - 1:
                energy_score_id = -log_sum_exp(logit, 1)
                energy_score_list = torch.cat((energy_score_list, energy_score_id), 0)

        elif sum_temp == num_classes * args.sample_number and epoch == args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0)
            
            energy_score_id = -log_sum_exp(logit, 1)
            incorrect_preds = torch.argmax(logit, 1) != target
            energy_ood_preds = energy_score_id > energy_threshold
            initial_oods = data[incorrect_preds & energy_ood_preds]
            
            if len(initial_oods) > 0:
                if isinstance(net, torch.nn.DataParallel):
                    ood_logits, ood_outputs = net.module.forward_virtual(initial_oods)
                else:
                    ood_logits, ood_outputs = net.forward_virtual(initial_oods)
                
                energy_score_for_fg = -log_sum_exp(logit, 1)
                energy_score_for_bg = -log_sum_exp(ood_logits, 1)
    
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).to(device), torch.zeros(len(ood_logits)).to(device)), -1)
                
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = F.cross_entropy(output1, labels_for_lr.long())
                
                kl_loss = F.kl_div(F.log_softmax(ood_logits, dim=1), torch.full_like(ood_logits, 1.0 / num_classes),
                                   reduction='batchmean', log_target=False)
            
        elif sum_temp == num_classes * args.sample_number and epoch > args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0)

            fgsm_ood = generate_fgsm_ood(args, data, target)
            
            if isinstance(net, torch.nn.DataParallel):
                sel_ood_logits, _ = net.module.forward_virtual(fgsm_ood)
            else:
                sel_ood_logits, _ = net.forward_virtual(fgsm_ood)
            
            energy_score_for_fg = -log_sum_exp(logit, 1)
            energy_score_for_bg = -log_sum_exp(sel_ood_logits, 1)
            
            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
            labels_for_lr = torch.cat((torch.ones(len(output)).to(device), torch.zeros(len(sel_ood_logits)).to(device)), -1)
            
            output1 = logistic_regression(input_for_lr.view(-1, 1))
            lr_reg_loss = F.cross_entropy(output1, labels_for_lr.long())
            
            kl_loss = F.kl_div(F.log_softmax(sel_ood_logits, dim=1), torch.full_like(sel_ood_logits, 1.0 / num_classes),
                               reduction='batchmean', log_target=False)
        
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1
        
        optimizer.zero_grad()
        loss = F.cross_entropy(logit, target) + args.lr_loss_weight * lr_reg_loss + args.kl_loss_weight * kl_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        if batch_idx % 300 == 0:
            print(f'Epoch {epoch+1}, LR Reg Loss={lr_reg_loss.item()}, KL Loss={kl_loss.item()} Total Loss={loss.item()}')
    
    if len(energy_score_list) != 0:
        energy_threshold = torch.quantile(energy_score_list, 0.95).item()
        print(f"Threshold for OOD detection: {energy_threshold:.4f} / len : {len(energy_score_list)}")
    
    return loss_avg, lr_reg_loss
    
#%%
def evaluate():
    net.eval()
    loss_avg = 0.0
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_avg += float(loss.data)
    
    loss_avg = loss_avg / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {loss_avg:.4f} | Test Accuracy: {accuracy:.2f}%')
    return loss_avg, accuracy

def main(args):
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loss, lr_reg_loss = train_EOD(epoch)
        test_loss, id_accuracy = evaluate()
        print('Epoch', epoch + 1, 'lr', optimizer.param_groups[0]['lr'],
              'LR Loss', round(lr_reg_loss.item(), 5), 'Total Train Loss', round(train_loss, 5),
              'ID Test Loss', round(test_loss, 5), 'ID Test Accuracy', round(id_accuracy, 2))
        
    model_path = f'{save_path}_epoch{epoch+1}.pt'
    print(f'Epoch {epoch+1} model saved at {model_path}')
    torch.save(net.state_dict(), model_path)

if __name__ == '__main__':
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    if 'CIFAR' in args.dataset:
        args.img_size, args.batch_size, args.test_bs = 32, 128, 200
        train_dataset, test_dataset, train_loader, test_loader, num_classes = CIFAR_dataloader(args)
    elif 'IMAGENET' in args.dataset:
        args.img_size, args.batch_size, args.test_bs = 224, 128, 200
        train_dataset, test_dataset, train_loader, test_loader, num_classes = IMAGENET_dataloader(args)

    ood_loaders = CIFAR_oodloader(args)
    svhn_loaders = ood_loaders['SVHN']
    places365_loaders = ood_loaders['PLACES365']

    if args.model == 'allconv':
        net = AllConvNet(num_classes)
    elif args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu))).to(device) if args.ngpu > 1 else net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    run_name = f'{args.mode}_{args.dataset}_{args.model}_seed{args.seed}_lr{args.lr_loss_weight}_kl{args.kl_loss_weight}_thres{args.threshold}_scale{args.scale}'
    save_path = os.path.join(args.save, args.dataset, run_name)
    print(run_name)
    
    energy_threshold = None
    weight_energy = torch.nn.Linear(num_classes, 1).to(device)
    torch.nn.init.uniform_(weight_energy.weight)
    data_dict = torch.zeros(num_classes, args.sample_number, args.penultimate_dim).to(device)
    number_dict = {i: 0 for i in range(num_classes)}
    eye_matrix = torch.eye(args.penultimate_dim, device=device)
    logistic_regression = torch.nn.Linear(1, 2).to(device)

    optimizer = torch.optim.SGD([
        {'params': net.parameters()},
        {'params': weight_energy.parameters()},
        {'params': logistic_regression.parameters()}],
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1, 1e-6 / args.learning_rate))
    
    main(args)