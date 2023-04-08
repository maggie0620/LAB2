# LAB2
EEG classification

## dataloader.py
    S4b_train = np.load('./lab2/lab2_dataset/train/S4b_train.npz')
    X11b_train = np.load('./lab2/lab2_dataset/train/X11b_train.npz')
    S4b_test = np.load('./lab2/lab2_dataset/test/S4b_test.npz')
    X11b_test = np.load('./lab2/lab2_dataset/test/X11b_test.npz')
 
## EEGNet.py 
FirstConv -> DepthwiseConv -> SeparableConv -> Classification
ELU alpha value = 0.15
Dropout probability = 0.4

## main.py
Num epochs = 150 
Batch size = 64
Learning rate = 0.001

Optimizer：Adam
Loss function：torch.nn.CrossEntropyLoss()
Weight decay = 0.005
