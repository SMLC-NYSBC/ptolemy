import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class LowMag_64x5_2ep(nn.Module):
    def __init__(self):
        super(LowMag_64x5_2ep, self).__init__()

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
        final = self.linear(x)
        # x = self.pooling(self.bn6(self.activation(self.layer6(x))))
        # x = self.output(x.reshape(-1, 128))
        return torch.sigmoid(final)

    def score_crops(self, preprocessed_crops):
        with torch.no_grad():
            batch = torch.tensor(np.array(preprocessed_crops)).unsqueeze(1).float().to(next(self.layer1.parameters()).device)
            return self.forward(batch).detach().cpu().numpy().flatten()
        #     scores = []
        #     for crop in preprocessed_crops:
        #         crop = torch.tensor(crop).unsqueeze(0).unsqueeze(0).float()
        #         scores.append(self.forward(crop).item())
        
        # return scores


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

    def get_mask(self, mm_image):
        with torch.no_grad():
            mm_image = torch.tensor(mm_image).unsqueeze(0).unsqueeze(0).float().to(next(self.last.parameters()).device)
            output = self.forward(mm_image).detach().cpu().numpy()[0, 0]
        return output


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

class Hole_Classifier_Multitask(nn.Module):
    def __init__(self, n_layers, n_filters, inpsize=300):
        super(Hole_Classifier_Multitask, self).__init__()
        self.inpsize = inpsize
        for i in range(n_layers):
            inpsize = (inpsize - 1) // 2
        
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(1, n_filters, 3, 1, bias=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm2d(n_filters))
        
        for _ in range(1, n_layers):
            self.convs.append(nn.Conv2d(n_filters, n_filters, 3, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(n_filters))
        
        self.pooling = nn.MaxPool2d(3, 2, padding=1)
        self.activation = nn.ReLU()
        
        self.final1 = nn.Conv2d(n_filters, 1, inpsize, 1, padding=0)
        self.final2 = nn.Conv2d(n_filters, 1, inpsize, 1, padding=0)
        self.final3 = nn.Conv2d(n_filters, 1, inpsize, 1, padding=0)
    
    def forward(self, x):
        # return self.forward_cropped(x)
        for conv, bn in zip(self.convs, self.bns):
            x = self.pooling(self.activation(bn(conv(x))))
            
        return self.final1(x), self.final2(x), self.final3(x)
    
    def get_activations(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pooling(self.activation(bn(conv(x))))
            
        return x

    def extract_features(self, batch):
        with torch.no_grad():
            batch = torch.tensor(batch).float().to(next(self.final1.parameters()).device)
            return self.get_activations(batch).squeeze().detach().cpu().numpy()

class BasicFixedDimModel(nn.Module):
    def __init__(self, n_layers, n_filters, inpsize=402):
        super(BasicFixedDimModel, self).__init__()
        for i in range(n_layers):
            inpsize = (inpsize - 1) // 2
        
        
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(1, n_filters, 3, 1, bias=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm2d(n_filters))
        
        for _ in range(1, n_layers):
            self.convs.append(nn.Conv2d(n_filters, n_filters, 3, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(n_filters))
        
        self.pooling = nn.MaxPool2d(3, 2, padding=1)
        self.activation = nn.ReLU()
        
        self.final = nn.Conv2d(n_filters, 1, inpsize, 1, padding=0)
    
    def forward(self, x):
        # return self.forward_cropped(x)
        for conv, bn in zip(self.convs, self.bns):
            x = self.pooling(self.activation(bn(conv(x))))
            
        return self.final(x)
    
    def forward_final(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pooling(self.activation(bn(conv(x))))
            
        return x, self.final(x)

    def score_batch(self, batch):
        with torch.no_grad():
            batch = torch.tensor(batch).float().to(next(self.final.parameters()).device)
            return torch.sigmoid(self.forward(batch)).flatten().detach().cpu().numpy()


        

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        blur = torchvision.transforms.GaussianBlur(5, 1.5)
        resize = torchvision.transforms.Resize(100)
        self.composed = torchvision.transforms.Compose([blur, resize])
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(36, 1)
        # self.fc2 = nn.Linear(128, 1)
        

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.maxpool5(x)
        
        # Flatten output for fully-connected layers
        x = x.view(x.size(0), -1)
        
        # Fully-connected layer
        x = self.fc1(x)
        # x = nn.functional.relu(x)
        # x = self.fc2(x)
        
        return x
    
    def score_batch(self, batch):
        batch = torch.tensor(batch).float()
        batch = self.composed(batch)
        with torch.no_grad():
            batch = batch.to(next(self.fc1.parameters()).device)
            return torch.sigmoid(self.forward(batch)).flatten().detach().cpu().numpy()