import torch.nn as nn
import torch
provNum, alphaNum, adNum = 38, 25, 35
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class Main_model(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(Main_model, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
#             nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
#             nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            small_basic_block(ch_in=128, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
#             nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(1, 13), stride=1, padding=[0,6]), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        
        self.connected = nn.Sequential(
            nn.Linear(class_num*88,128),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=128+self.class_num, out_channels=self.class_num, kernel_size=(1, 88), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )
        self.classifier1 = nn.Linear(self.class_num,provNum)
        self.classifier2 = nn.Linear(self.class_num,alphaNum)
        self.classifier3 = nn.Linear(self.class_num,adNum)
        self.classifier4 = nn.Linear(self.class_num,adNum)
        self.classifier5 = nn.Linear(self.class_num,adNum)
        self.classifier6 = nn.Linear(self.class_num,adNum)
        self.classifier7 = nn.Linear(self.class_num,adNum)

    def forward(self, x):
        
        x = self.backbone(x)
      
        pattern = x.flatten(1,-1)
        
        pattern = self.connected(pattern)
      
        width = x.size()[-1]
        pattern = torch.reshape(pattern,[-1,128,1,1])
        pattern = pattern.repeat(1,1,1,width)
       
        x = torch.cat([x,pattern],dim=1)
       
        x = self.container(x)
      
        logits = x.squeeze(2)
        out = logits.squeeze(2)
        y0 = self.classifier1(out)
        y1 = self.classifier2(out)
        y2 = self.classifier3(out)
        y3 = self.classifier4(out)
        y4 = self.classifier5(out)
        y5 = self.classifier6(out)
        y6 = self.classifier7(out)
        return [y0, y1, y2, y3, y4, y5, y6]

def build_model(class_num=66, dropout_rate=0.5):

    Net = Main_model(class_num, dropout_rate)
    
    return Net

#     if phase == "train":
#         return Net.train()
#     else:
#         return Net.eval()

if __name__ == "__main__":
#     from torchsummary import summary
    model = build_model(75,0.5)
#     summary(model, (3,24,94), device="cpu")
    x = torch.rand[128, 3, 24, 94]
    predict = model(x)
    print(x.size)