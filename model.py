import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
    
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)
        
    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x
    
class SSDPlus(nn.Module):

    def __init__(self, num_classes, bboxes, pretrain=True):
        super(SSDPlus, self).__init__()

        self.num_classes = num_classes

        self.L2Norm = L2Norm(512, 20)

        if pretrain:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
        
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.extra_layers = nn.ModuleList([
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
        ])

        self.regression_headers = nn.ModuleList([
            nn.Conv2d(512, bboxes[0] * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, bboxes[1] * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, bboxes[2] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, bboxes[3] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, bboxes[4] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, bboxes[5] * 4, kernel_size=3, padding=1)
        ])
        self.classification_headers = nn.ModuleList([
            nn.Conv2d(512, bboxes[0] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, bboxes[1] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, bboxes[2] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, bboxes[3] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, bboxes[4] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, bboxes[5] * num_classes, kernel_size=3, padding=1)
        ])



    def forward(self, x):

        features = []

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 5:
                s = self.L2Norm(x)
                features.append(s)
            elif i == 6:
                features.append(x)

        for i, layer in enumerate(self.extra_layers):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                features.append(x)

        locations = []
        confidences = []
        for s, rh, ch in zip(features, self.regression_headers, self.classification_headers):
            locations.append(rh(s).permute(0, 2, 3, 1).contiguous())
            confidences.append(ch(s).permute(0, 2, 3, 1).contiguous())

        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)
        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        locations = locations.view(locations.size(0), -1, 4)
        confidences = confidences.view(confidences.size(0), -1, self.num_classes)
        return locations, confidences
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, reduction='mean'):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if reduction == 'mean':
            return F_loss.mean()
        elif reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def hard_negtives(logits, labels, pos, neg_radio, loss_fn):
    num_batch, num_anchors, num_classes = logits.shape
    logits = logits.view(-1, num_classes)
    labels = labels.view(-1)

    if loss_fn is None:
        losses = F.cross_entropy(logits, labels, reduction='none')
    else:
        losses = loss_fn(logits, labels, reduction='none')

    losses = losses.view(num_batch, num_anchors)

    losses[pos] = 0

    
    loss_idx = losses.argsort(1, descending=True)
    rank = loss_idx.argsort(1) 

    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(neg_radio*num_pos, max=pos.shape[1]-1) #(batch, 1)
    neg = rank < num_neg.expand_as(rank)
    return neg
    
class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes=10, neg_radio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_radio = neg_radio
        self.loss_fn = FocalLoss()
    
    def forward(self, pred_loc, pred_label, gt_loc, gt_label):
        pos_idx = gt_label > 0
        pos_loc_idx = pos_idx.unsqueeze(2).expand_as(pred_loc)
        pred_loc_pos = pred_loc[pos_loc_idx].view(-1, 4)
        gt_loc_pos = gt_loc[pos_loc_idx].view(-1, 4)

        loc_loss = F.smooth_l1_loss(pred_loc_pos, gt_loc_pos, reduction='sum')

        
        logits = pred_label.detach()
        labels = gt_label.detach()
        neg_idx = hard_negtives(logits, labels, pos_idx, self.neg_radio, self.loss_fn) #neg (batch, n)

        pos_cls_mask = pos_idx.unsqueeze(2).expand_as(pred_label)
        neg_cls_mask = neg_idx.unsqueeze(2).expand_as(pred_label)

        conf_p = pred_label[(pos_cls_mask+neg_cls_mask).gt(0)].view(-1, self.num_classes)
        target = gt_label[(pos_idx+neg_idx).gt(0)]

        cls_loss = F.cross_entropy(conf_p, target, reduction='sum')
        N = pos_idx.long().sum()

        loc_loss /= N
        cls_loss /= N


        return loc_loss, cls_loss