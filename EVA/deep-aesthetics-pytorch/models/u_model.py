import torch.nn as nn
from models.mv2 import mobile_net_v2
from models.mv2 import Backbone_1
class NIMA(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMA, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])
        #base_model = Backbone_1("resnet18",pretrained_base_model)
        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.75),
            #nn.Linear(512, 12),
            nn.Linear(1280,5),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

