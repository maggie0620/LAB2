import torch.nn as nn

# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        # Layer 1 (FirstConv)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, eps = 1e-05, momentum=0.1, affine = True, track_running_stats=True)
        
        # Layer 2 (DepthwiseConv)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, eps = 1e-05, momentum=0.1, affine = True, track_running_stats=True)
        self.elu2 = nn.ELU(alpha=0.15)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.dropout2 = nn.Dropout(p=0.4)
        
        # Layer 3 (SeparableConv)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, eps = 1e-05, momentum=0.1, affine = True, track_running_stats=True)
        self.elu3 = nn.ELU(alpha=0.15)
        self.pooling3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.dropout3 = nn.Dropout(p=0.4)
        
        # Classify
        self.fc1 = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)
        
        # FC Layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x