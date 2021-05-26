import copy
import torch.nn as nn
import torch
import math

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

class FRM(nn.Module):
    def __init__(self, p, kernel_size=7):
        super(FRM, self).__init__()
        padding = 3
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.p = p
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        residual = x
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avgout, maxout], dim=1)
        y = self.conv(y)
        y = self.bn(y)
        y = self.sigmoid(y)
        y_mask = y > self.p
        y = y.masked_fill(y_mask, 0)
        y = y.masked_fill(~y_mask, 1)
        y = y.expand_as(x)
        out = x * y

        out += residual
        out = self.relu(out)

        return out


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, ibn=False, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, last_stride, block, layers, frozen_stages=-1, num_classes=1000, dataset="m"):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.dataset = dataset
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.frm1 = FRM(p=0.85)
        self.frm2 = FRM(p=0.9)
        self.frm3 = FRM(p=0.95)
        self.layer31 = copy.deepcopy(self.layer3)
        self.layer41 = copy.deepcopy(self.layer4)
        if self.dataset == 'd':
            self.layer42 = copy.deepcopy(self.layer4)
        # if the dataset is DukeMTMC_reID, setting layer42 will have better performance
        self.conv1.weight.data.normal_(0, math.sqrt(2. / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = False
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, stride, downsample, ibn=ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, ibn=ibn))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.frm1(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.layer31(x1)
        x2 = self.frm2(x1)
        x1 = self.layer41(x1)
        if self.dataset == 'd':
            x2 = self.layer42(x2)
        else:
            x2 = self.layer41(x2)
        x3 = self.frm3(x2)

        return x, x1, x2, x3

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def se_resnet50(last_stride, num_classes=1000, dataset="m", **kwargs):
    model = ResNet(last_stride, SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, dataset=dataset, **kwargs)
    return model


def se_resnet101(last_stride, num_classes=1000, dataset="m", **kwargs):
    model = ResNet(last_stride, SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, dataset=dataset, **kwargs)
    return model
