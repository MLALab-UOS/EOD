import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BinarySupConLoss(nn.Module):
    def __init__(self, temperature=0.1, device="cuda"):
        super(BinarySupConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, id_feature, ood_feature):
        num_samples = min(id_feature.size(0), ood_feature.size(0))
        id_feature = id_feature[torch.randperm(id_feature.size(0))[:num_samples]]
        ood_feature = ood_feature[torch.randperm(ood_feature.size(0))[:num_samples]]

        id_labels = torch.ones(num_samples, dtype=torch.long).to(self.device)
        ood_labels = torch.zeros(num_samples, dtype=torch.long).to(self.device)

        features = torch.cat([id_feature, ood_feature], dim=0)  # (2N, D)
        labels = torch.cat([id_labels, ood_labels], dim=0).view(-1, 1) # (2N,)
        
        features = F.normalize(features, dim=1) # Normalize를 해야하나 ?
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # (2N, 2N)

        mask = torch.eq(labels, labels.T).float().to(self.device)
        self_mask = torch.eye(mask.shape[0], dtype=torch.float32).to(self.device)
        mask = mask - self_mask

        exp_sim = torch.exp(similarity_matrix) * (1 - self_mask)
        denom = exp_sim.sum(dim=1, keepdim=True)  # (2N, 1)

        pos_exp = exp_sim * mask  # (2N, 2N)
        numerator = pos_exp.sum(dim=1, keepdim=True)

        supcon_loss = -torch.log((numerator + 1e-8) / (denom + 1e-8)).mean()

        return supcon_loss

class CompLoss(nn.Module):
    def __init__(self, n_cls, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.n_cls = n_cls
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        device = torch.device('cuda')

        proxy_labels = torch.arange(0, self.n_cls).to(device)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, proxy_labels.T).float().to(device)

        # compute logits
        anchor_feature = features
        contrast_feature = prototypes / prototypes.norm(dim=-1, keepdim=True)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss

class DispLoss(nn.Module):
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1, cifar=True):
        super(DispLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(self.args.n_cls,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.init_class_prototypes(if_cifar=cifar)

    def forward(self, features, labels):

        prototypes = self.prototypes
        num_cls = self.args.n_cls
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()


        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self, if_cifar):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.args.n_cls,self.args.feat_dim).cuda()
            for i, (input, target) in enumerate(self.loader):
                input, target = input.cuda(), target.cuda()
                if if_cifar:
                    features = self.model(input)
                else:
                    _, _, features = self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.n_cls):
                prototypes[cls] /=  prototype_counts[cls]
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes

#####
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class KLUniformLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLUniformLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits):
        _, num_classes = logits.shape
        log_probs = F.log_softmax(logits, dim=1)
        uniform_dist = torch.full_like(logits, 1.0 / num_classes)
        kl_loss = F.kl_div(log_probs, uniform_dist, reduction=self.reduction, log_target=False)
        return kl_loss