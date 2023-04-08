# LAB2
EEG classification
## 建立model、參數設定
Num epochs = 150 
Batch size = 64
Learning rate = 0.001 
Optimizer：Adam
Loss function：torch.nn.CrossEntropyLoss()
Weight decay = 0.005
 
## EEGNet介紹
Alpha value = 0.15
Dropout probability = 0.4
 
## ELU function介紹：
alpha value控制影響程度、p控制Dropout數量
