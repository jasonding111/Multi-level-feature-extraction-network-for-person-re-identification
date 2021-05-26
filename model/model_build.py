import torch
import torch.nn as nn
from losses.circle_loss import CircleLoss
from model.se_resnet import se_resnet50, se_resnet101
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                                     (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(p) + \
               ', ' + 'eps=' + str(self.eps) + ')'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MFEN(nn.Module):
    def __init__(self, num_classes,dataset):
        super(MFEN, self).__init__()
        last_stride = 1
        self.in_planes = 2048
        self.backbone = se_resnet50(last_stride, num_classes, dataset)
        pretrain = True
        model_path = './checkpoint/resnet50-19c8e357.pth'
        if pretrain:
            self.backbone.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.gap = GeM()
        self.gap1 = GeM()
        self.gap2 = GeM()
        self.gap3 = GeM()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap1 = nn.AdaptiveAvgPool2d(1)
        # self.gap2 = nn.AdaptiveAvgPool2d(1)
        # self.gap3 = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.classifier = CircleLoss(self.in_planes, self.num_classes, s=30, m=0.4)
        self.classifier1 = CircleLoss(self.in_planes, self.num_classes, s=30, m=0.4)
        self.classifier2 = CircleLoss(self.in_planes, self.num_classes, s=30, m=0.4)
        self.classifier3 = CircleLoss(self.in_planes, self.num_classes, s=30, m=0.4)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

    def forward(self, x, label=None):
        x, x1, x2, x3 = self.backbone(x)

        global_feat = self.gap(x)
        global_feat1 = self.gap1(x1)
        global_feat2 = self.gap2(x2)
        global_feat3 = self.gap3(x3)

        global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat1 = global_feat1.view(global_feat1.shape[0], -1)
        global_feat2 = global_feat2.view(global_feat2.shape[0], -1)
        global_feat3 = global_feat3.view(global_feat3.shape[0], -1)

        feat = self.bottleneck(global_feat)
        feat1 = self.bottleneck1(global_feat1)
        feat2 = self.bottleneck1(global_feat2)
        feat3 = self.bottleneck1(global_feat3)
        if self.training:
            cls_score = self.classifier(feat, label)
            cls_score1 = self.classifier1(feat1, label)
            cls_score2 = self.classifier2(feat2, label)
            cls_score3 = self.classifier3(feat3, label)

            return cls_score, global_feat, cls_score1, cls_score2, cls_score3
        else:
            return torch.cat((feat, feat1, feat2, feat3), 1)


def build_model(num_class, dataset):
    model = MFEN(num_class, dataset)
    return model
