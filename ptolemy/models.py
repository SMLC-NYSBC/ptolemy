import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LowMag_64x5_2ep(nn.Module):
    def __init__(self):
        super(Model_64_64_64_64_64, self).__init__()

        self.pooling = nn.MaxPool2d(3, 2, padding=1)
        self.activation = nn.ReLU()

        self.layer1 = nn.Conv2d(1, 64, 5, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer2 = nn.Conv2d(64, 64, 5, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 64, 3, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.layer5 = nn.Conv2d(64, 64, 3, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.linear = nn.Conv2d(64, 1, 6, 1, padding=0)
        # self.layer4 = nn.Conv2d(256, 128, 3, 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.layer5 = nn.Conv2d(128, 64, 3, 1, bias=False)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.layer6 = nn.Conv2d(64, 32, 3, 1, bias=False)
        # self.bn6 = nn.BatchNorm2d(32)
        # self.output = nn.Linear(128, 1, bias=True)
    
    def forward(self, x):
        x = self.pooling(self.bn1(self.activation(self.layer1(x))))
        x = self.pooling(self.bn2(self.activation(self.layer2(x))))
        x = self.pooling(self.bn3(self.activation(self.layer3(x))))
        x = self.pooling(self.bn4(self.activation(self.layer4(x))))
        x = self.pooling(self.bn5(self.activation(self.layer5(x))))
        x = self.linear(x)
        # x = self.pooling(self.bn6(self.activation(self.layer6(x))))
        # x = self.output(x.reshape(-1, 128))
        return x
    
class Wrapper:
    def __init__(self, model):
        self.model = model
        self.cuda = False
    
    # Might need to fix this later
    def to_cuda(self):
        self.model.cuda()
        self.cuda = True

    def to_cpu(self):
        self.model.cpu()
        self.cuda = False
    
    def forward_cropset(self, cropset):
        # Check cropset sizes
        sizes = set()
        for crop in cropset.crops:
            sizes.add(crop.shape)
        if len(sizes) == 1:
            batch = torch.tensor(cropset.crops).unsqueeze(0).unsqueeze(0).float()
            return self.forward_batch(batch)
        else:
            results = []
            for crop in cropset.crops:
                results.append(self.forward_single_scalarout(crop))
            return np.array(results)

    def forward_single(self, image):
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
        if self.cuda:
            image = image.cuda()
        output = self.model.forward(image).detach().cpu().numpy()[0, 0]
        return output
    
    def forward_single_scalarout(self, image):
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
        if self.cuda:
            image = image.cuda()
        output = self.model.forward(image).item()
        return output
        
    def forward_batch(self, batch):
        if self.cuda:
            batch = batch.cuda()
        
        output = self.model.forward(batch).detach().cpu().numpy().flatten()
        return output


class BasicUNet(nn.Module):    
    def __init__(self, n_channels, depth):
        super(BasicUNet, self).__init__()
        
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        self.n_channels = n_channels
        self.depth = depth
        prev_channels = 1

        self.down_path.append(DownBlock(1, n_channels))
        for _ in range(1, depth):
            self.down_path.append(DownBlock(n_channels, n_channels))

        for _ in range(1, depth):
            self.up_path.append(UpBlock(n_channels*2, n_channels))
        self.last = nn.Conv2d(n_channels, 1, 3, padding=1)

    def forward(self, x):
        blocks = [x]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != self.depth - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activ='relu'):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if activ == 'relu':
            self.activ = nn.ReLU()
        # TODO more activation functions

    def forward(self, x):
        return self.activ(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activ='relu', upsample='nearest'):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if activ == 'relu':
            self.activ = nn.ReLU()

        self.upsample_mode = upsample
        # TODO more activation functions

    def forward(self, x, skip):
        x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode=self.upsample_mode)
        x = torch.cat([x, skip], 1)
        return self.activ(self.conv(x))

class AveragePoolModel(nn.Module):
    def __init__(self, n_layers, n_filters):
        super(AveragePoolModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(1, n_filters, 5, 1, bias=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm2d(n_filters))
        
        for _ in range(1, n_layers):
            self.convs.append(nn.Conv2d(n_filters, n_filters, 3, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(n_filters))
        
        self.pooling = nn.MaxPool2d(3, 2, padding=0)
        self.activation = nn.ReLU()
        
        self.final = nn.Linear(n_filters, 1)
        
    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pooling(self.activation(bn(conv(x))))
        x = torch.mean(x, [2, 3])
        output = self.final(x).squeeze(0)
        return output
        
#         results = []
        
#         for img, this_shape in zip(x, shape):
#             result = self.forward_cropped(self.crop(img, this_shape))
#             results.append(result)
            
#         return torch.cat(results)
            
#     def crop(self, x, shape):
#         return x[:shape[0], :shape[1]]
        
#     def forward_cropped(self, x):
#         x = x.unsqueeze(0)
#         for conv, bn in zip(self.convs, self.bns):
#             x = self.pooling(self.activation(bn(conv(x))))
            
#         x = torch.mean(x, [2, 3])
#         output = self.final(x).squeeze(0)
#         return output