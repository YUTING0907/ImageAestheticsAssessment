import torch.nn as nn
import torch
from models.mv2 import cat_net


def SelfAttentionMap(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = cat_net()
        self.base_model = base_model
        for p in self.parameters():
            p.requires_grad = False
        
        #self.avg = nn.AdaptiveAvgPool2d((10, 1))
        self.head = nn.Sequential(
            nn.ReLU(),
            #nn.Dropout(p=0.75),
            nn.Linear(5376,2048),
            nn.ReLU(),
            #nn.Dropout(p=0.75),
            #nn.Linear(2048,12),
            #nn.Linear(4608,12), #resnet
            #nn.Softmax(dim=1)
        )

        self.hidden1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2048,12),
                )
        self.hidden2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1280,512)
                )
        self.hidden3 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(512,128)
                )
        self.hidden4 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128,32)
                )
        self.head2 = nn.Sequential(
                nn.ReLU(),
                #nn.Dropout(p=0.75),
                nn.Linear(32,12)
                )

    def forward(self, x):

        x1, x2 = self.base_model(x)
        print(x1.shape,x2.shape)
        x = torch.cat([x1,x2],1)
        x = self.head(x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        #x = self.hidden3(x)
        #x = self.hidden4(x)
        #x = self.head2(x)
        
        return x

#
# if __name__=='__main__':
#     model = NIMA()
#     x = torch.rand((16,3,224,224))
#     out = model(x)
