import torch
from torch import nn
import torchvision
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np
import torch.nn.functional as F

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Reward_NN(nn.Module):
    def __init__(self, in_channel=4096):
        super().__init__()
        self.in_channel = in_channel
        self.linear1 = nn.Linear(in_channel, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 8)
        self.linear7 = nn.Linear(8, 1)  # final reward is a value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        out = F.relu(self.linear5(out))
        out = F.relu(self.linear6(out))
        out = torch.tanh(self.linear7(out))
        return out

class Get_Reward:
    def __init__(self):
        self.reward_hist = []

    def send_reward(self, screen_feature):
        reward_nn = Reward_NN()
        reward_nn.eval()
        reward = reward_nn(torch.Tensor(screen_feature ))
        return reward.detach().cpu().numpy()


class FeatureMap:
    def __init__(self, size=224):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor()
        ])


    def FeatureExtract(self, screen_shot):
        Image_data = self.transform(screen_shot)
        vgg16.classifier = vgg16.classifier[:4]
        vgg16_ = vgg16.to(device)
        Image_data = Image_data.to(device)
        pred = vgg16_(Image_data.unsqueeze(dim=0))
        feature = np.array(pred)
        return feature


class Feature2Act(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_shape[1], out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        # we seperate the linear network into two parts, one part is
        # for value and another is advantage stream.
        # Here a represents advantage stream and b represents value.
        self.fc3a = nn.Linear(in_features=1024, out_features=128)
        self.fc3b = nn.Linear(in_features=1024, out_features=128)
        self.fc4a = nn.Linear(in_features=128, out_features=17)
        self.fc4b = nn.Linear(in_features=128, out_features=1)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        xa = F.relu(self.fc3a(x))
        xa = F.relu(self.fc4a(xa))
        xb = F.relu(self.fc3b(x))
        xb = F.relu(self.fc4b(xb))
        xh = torch.add(xa, xb)
        return xh









