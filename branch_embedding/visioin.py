## By Preeti
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,flag = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or flag == 0:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,flag =1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or flag == 0:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,branch, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        branch = str(branch)[::-1]
        branch = int(branch)
        
        b = branch%(pow(10,num_blocks[0]))
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,br = b)
        branch = branch//(pow(10,num_blocks[0]))
        
        b = branch%(pow(10,num_blocks[1]))        
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,br = b)
        branch = branch//(pow(10,num_blocks[1]))
        
        b = branch%(pow(10,num_blocks[2]))        
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,br = b)
        branch = branch//(pow(10,num_blocks[2]))
        
        b = branch%(pow(10,num_blocks[3]))        
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,br = b)
        
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,br):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,flag = br%10))
            self.in_planes = planes * block.expansion
            br = br//10
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(branch):
    return ResNet(BasicBlock, [2, 2, 2, 2],branch)


def ResNet34(branch):
    return ResNet(BasicBlock, [3, 4, 6, 3],branch)


def ResNet50(branch):
    return ResNet(Bottleneck, [3, 4, 6, 3],branch)


def ResNet101(branch):
    return ResNet(Bottleneck, [3, 4, 23, 3],branch)


def ResNet152(branch):
    return ResNet(Bottleneck, [3, 8, 36, 3],branch)


#latency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#https://deci.ai/blog/measure-inference-time-deep-neural-networks/
#warm-up gpu
lr=0.1
model1 = ResNet50(branch = 1111111111111111)
model1.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=lr,
                  momentum=0.9, weight_decay=5e-4)
model1.load_state_dict(torch.load('/home/preeti/Transformers/ebm/resnet50_94pc.pt'),strict=False)
model1.eval()
dummy_input = torch.randn(1,3,32,32, dtype=torch.float).to(device)

#GPU-WARM-UP
for _ in range(10):
    _ = model1(dummy_input)

#using branches that only work and hava acc>50
branches = [1111111111111100, 1111111111111010, 1111111111111001, 1111111111110110, 1111111111110101, 1111111111110011, 1111111111101101, 
            1111111111101011, 1111111111011101, 1111111111011011, 1111111011111110, 1111111011111101, 1111111011111011, 1111111011110111, 
            1111111011101111, 1111111011011111, 1110111111111110, 1110111111111101, 1110111111111011, 1110111111110111, 1110111111101111, 
            1110111111011111, 1110111011111111, 111111111111110, 111111111111101, 111111111111011, 111111101111111, 111011111111111, 
            1111111111111111]
#branches = [1111111111111111]

#print(branches)

#ranking branches as per latency
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.01
a =[]
d = {}
for b in branches:
    #model1 = ResNet152(branch = b)
    model1 = ResNet50(branch = b)
    model1.to(device)
    #print("Branch:",b)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model1.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    model1.load_state_dict(torch.load('/home/preeti/Transformers/ebm/resnet50_94pc.pt'),strict=False)
    
    model1.eval()  
    
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1
    timings=np.zeros((repetitions,1))
    
    
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model1(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        #print("Time required in millisec:",mean_syn)
        a.append(mean_syn)
        d[b] = mean_syn


d_sorted= sorted(d.items(),key=lambda x: x[1])
#print("SORTING DONE, branches in increasing order of latency:")
# for i in range (len(d_sorted)):
#     print(f"Branch:{d_sorted[i][0]} Latency:{d_sorted[i][1]}")

l_50 = []
for i in d_sorted:
    l_50.append(i[0])

print(f"There are {len(l_50)} number of branches")


#create dataset
import torchvision 
image_dataloader_train = []
label_dataloader_train = []
i=0

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

# l = len(testloader)
# print(f"length of testloader {l}")

# images = []
# labels = []

# for image, label in testloader:
#     images.append(image)
#     labels.append(label.item())
#     #print("Label:",label)


# #choose first 5 of each label
# from collections import defaultdict, Counter

# items = Counter(labels).keys()
# print("No of unique items in the dataset are:", len(items))

# # Sample lists of features and labels
# features = images
# labels = labels

# labels_train = []
# labels_test = []
# # Create a dictionary to store selected features for each unique label
# selected_features = defaultdict(list)
# selected_features_test = defaultdict(list)

# # Iterate through features and labels
# for feature, label in zip(features, labels):
#     # Check if we have already selected 50 features for this label
#     if len(selected_features[label]) < 500:
#         selected_features[label].append(feature)
#         labels_train.append(label)
#     elif len(selected_features_test[label])<500:
#         selected_features_test[label].append(feature)
#         labels_test.append(label)



# # Flatten the selected features dictionary into a list
# final_selected_features = [feature for sublist in selected_features.values() for feature in sublist]
# final_selected_features_test = [feature for sublist in selected_features_test.values() for feature in sublist]
# print(f"There are {len(final_selected_features)} datapoints in train and {len(labels_train)} labels")
# print(f"Size of each datapoint is {final_selected_features[0].size()}")
# print(f"Size of label is {len(labels_train)}")

# print(f"There are {len(final_selected_features_test)} datapoints in test and {len(labels_test)} labels")
# print(f"Size of each datapoint is {final_selected_features_test[0].size()}")
# print(f"Size of label is {len(labels_test)}")


# #image feature extractor
# data = torch.stack(final_selected_features).to(device)
# #data = torch.squeeze(data, 1)
# data_test = torch.stack(final_selected_features_test).to(device)
# #data_test = torch.squeeze(data_test,1)

# print("Data size:",data.size())
# print("Data_test size:",data_test.size())

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet_feat(nn.Module):
    def __init__(self, block, layers,branch=0, num_classes = 10):
        super(ResNet_feat, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        branch_string = str(branch)
        reverse = "".join(reversed(branch_string))
        b = int(reverse)
        #c = b%10
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1,br=b)
        b = int(b/np.power(10,layers[0]))
        #c = b%10
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2,br=b)
        b = int(b/np.power(10,layers[1]))
        #c = b%10
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2,br=b)
        b = int(b/np.power(10,layers[2]))
        #c = b%10
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2,br=b)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks,br, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes or br==0:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        if br == 0:
            layers.append(downsample)
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            if br%10 == 0:
                downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes))
                layers.append(downsample)
            else:
                layers.append(block(self.inplanes, planes))
            br = int(br/10)

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
num_classes = 10
num_epochs = 50
batch_size = 64
learning_rate = 0.01

image_feat = ResNet_feat(ResidualBlock, [3, 4, 6, 3],branch = 1111111111111111).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(image_feat.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
image_feat.to(device)
image_feat.load_state_dict(torch.load('/home/preeti/Transformers/ebm/branch_resnet1.pt'),strict=False)
image_feat.eval()


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64) # change as per si
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,512)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

# Initialize the model
mlp = MLP()
mlp = mlp.to(device)
a =np.zeros(100)
best_b = np.zeros(100)
#print(best_b)
#branches = [1111111111111111,111111111111011,1101111111101111]
acc =[]
dataset = []
dataset_test = []
acc_test = []
for b in tqdm(branches):
    model1 = ResNet50(branch = b)    
    model1.to(device)
    #print("Branch:",b)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model1.parameters(), lr=lr,momentum=0.9, weight_decay=5e-4)
    model1.load_state_dict(torch.load('/home/preeti/Transformers/ebm/resnet50_94pc.pt'),strict=False)    
    model1.eval()
    correct = 0

    #### testing model capacity
    # with torch.no_grad():
    #     total = 0
    #     for inputs, targets in testloader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model1(inputs)
    #         #loss = criterion(outputs, targets)

    #         #test_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()

    #         # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #         #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # # Save checkpoint.
    # acc = 100.*correct/total
    # print("Accuracy:",acc)
    #############################################################################

    with torch.no_grad():
        #choosing first 5000 samples for training
        total = 0
        for images, targets in tqdm(testloader):
            images, targets = images.to(device), targets.to(device)
            outputs = model1(images)
            #loss = criterion(outputs, targets)

            #test_loss += loss.item()
            _, predicted = outputs.max(1)
            i_feat = image_feat(images)
            i_feat = torch.squeeze(i_feat)
            res = [int(x) for x in str(b)]
            if len(res)<16:
                break
            b1 = torch.Tensor(res)
            b1 = b1.to(device)
            branch_feat = mlp(b1)
            temp = torch.cat((i_feat,branch_feat))
                        
            #print(f"Predicted:{predicted.item()} Label:{labels}")
            if total>=9920:
                break
            if total<=9599:
                dataset.append(temp)
                if predicted.item() == targets:
                    acc.append(1)
                    correct =+1
                else:
                    acc.append(0)
                #del images, targets, outputs
            else:
                dataset_test.append(temp)
                if predicted.item() == targets:
                    acc_test.append(1)
                    correct =+1
                else:
                    acc_test.append(0)
                #del images, targets, outputs

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            del images, targets, outputs


            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        #accu = 100.*correct/total
        #print("Accuracy:",accu)
        print("Correct:",correct)
        print("Length of dataset:", len(dataset))
        print("Length of dataset test:", len(dataset_test))






#     #original code for 10 classes
#     with torch.no_grad():
#         acc_k = []             
#         labels_idx=0

#         for images in data: #for training
#             labels = labels_train[labels_idx] 
#             labels = torch.tensor(labels)
            
#             labels_idx = labels_idx+1
                
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model1(images)
#             _, predicted = outputs.max(1)
#             #_, predicted = torch.max(outputs.data, 1)
#             i_feat = image_feat(images)
#             i_feat = torch.squeeze(i_feat)
#             res = [int(x) for x in str(b)]
#             if len(res)<16:
#                 break
#             b1 = torch.Tensor(res)
#             b1 = b1.to(device)
#             branch_feat = mlp(b1)
#             temp = torch.cat((i_feat,branch_feat))
#             dataset.append(temp)            
#             #print(f"Predicted:{predicted.item()} Label:{labels}")
#             if predicted.item() == labels:
#                 acc.append(1)
#                 correct =+1
#             else:
#                 acc.append(0)
#             del images, labels, outputs
             
#         labels_idx=0
#         print("correct:",correct)


#         for images in data_test: #for test
#             labels = labels_test[labels_idx] 
#             labels = torch.tensor(labels)
            
#             labels_idx = labels_idx+1
                
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model1(images)
#             _, predicted = torch.max(outputs.data, 1)
#             i_feat = image_feat(images)
#             i_feat = torch.squeeze(i_feat)
#             res = [int(x) for x in str(b)]
#             if len(res)<16:
#                 break
#             b1 = torch.Tensor(res)
#             b1 = b1.to(device)
#             branch_feat = mlp(b1)
#             temp = torch.cat((i_feat,branch_feat))
#             dataset_test.append(temp)            
#             #print(f"Predicted:{predicted.item()} Label:{labels}")
#             if predicted.item() == labels:
#                 acc_test.append(1)
#                 #print("Correct")
#             else:
#                 acc_test.append(0)
#             del images, labels, outputs


# print(f"length of dataset{len(dataset)} and len of acc {len(acc)}")
# print(f"length of dataset_test{len(dataset_test)} and len of acc {len(acc_test)}")

d_train = []
for i in dataset:
    d_train.append(i.cpu().numpy())

d_test = []
for i in dataset_test:
    d_test.append(i.cpu().numpy())
    
X_test = np.asarray(d_test)
X_train = np.asarray(d_train)
print("Dataset train shape:",X_train.shape)
print("Dataset test shape:",X_test.shape)

# #https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Copied from: https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    
    p = torch.sigmoid(inputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


X = []
X.extend(X_train)
X.extend(X_test)
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
X = scaler.fit_transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print("NO scalar transform")
##last change does not work

y_train = acc
y_test = acc_test

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.1

## train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


#train_data = TrainData(torch.FloatTensor(X_train),torch.FloatTensor(y_train))
train_data = TrainData(torch.as_tensor(X_train),torch.as_tensor(y_train))


## test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
#test_data = TestData(torch.FloatTensor(X_test))
test_data = TestData(torch.as_tensor(X_test))

#test_data = TestData(torch.FloatTensor(X))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # # Number of input features is 12.
        # #self.layer_1 = nn.Linear(3073, 1024)
        # self.layer_2 = nn.Linear(1024, 512)
        # self.layer_3 = nn.Linear(512,128)
        self.layer_3_small = nn.Linear(1024,128)
        self.layer_4 = nn.Linear(128,64)        
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)
        
        
    def forward(self, inputs):
        # #x = self.relu(self.layer_1(inputs))
        # #x = self.batchnorm1(x)
        # x = self.relu(self.layer_2(inputs))
        # x = self.batchnorm2(x)
        
        x = self.relu(self.layer_3_small(inputs))
        x = self.dropout(x)
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)        
        
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
model = BinaryClassification()
model.to(device)
print(model)
#criterion = nn.BCEWithLogitsLoss()
#pos_weight = torch.randint(1, 100, (1,)).float()
#pos_weight = torch.tensor(4.0)    
pos_weight = torch.tensor(0.25)    
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#model.load_state_dict(torch.load('/home/preeti/Transformers/ebm/final_upto_two_blk_152.pt'),strict=False)
#model.eval()

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

loss_list = []
accuracy = []
loss_list_test = []
accuracy_test = []


for e in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_batch = y_batch*1.0
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        #loss = criterion(y_pred, y_batch.unsqueeze(1))
        loss = sigmoid_focal_loss(y_pred, y_batch.unsqueeze(1))
        #acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        #f1_acc(actual,predicted)
        
        y_pred = torch.round(torch.sigmoid(y_pred))
        predicted = y_pred.cpu().detach().numpy()
        actual = y_batch.unsqueeze(1).cpu().detach().numpy()
        acc = f1_score(actual,predicted)

        #loss.backward()
        #for focal loss
        loss.sum().backward()
        optimizer.step()
        
        #epoch_loss += loss.item()
        epoch_loss += loss.sum().item()
        #epoch_acc += acc.item()
        epoch_acc += acc
    
    epoch_test_loss = 0
    epoch_test_acc = 0

    with torch.no_grad():   
        model.eval()
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
        
            y_pred = model(X_batch)

            loss = sigmoid_focal_loss(y_pred, y_batch.unsqueeze(1))

        
            #loss = criterion(y_pred, y_batch.unsqueeze(1))
            #acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            #f1_acc(actual,predicted)
            
            y_pred = torch.round(torch.sigmoid(y_pred))
            predicted = y_pred.cpu().detach().numpy()
            actual = y_batch.unsqueeze(1).cpu().detach().numpy()
            acc = f1_score(actual,predicted)
        
            #epoch_test_loss += loss.item()
            epoch_test_loss += loss.sum().item()            
            #epoch_test_acc += acc.item()
            epoch_test_acc += acc

    print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | Train F1: {epoch_acc/len(train_loader):.3f} | Test Loss: {epoch_test_loss/len(test_loader):.5f} | Test F1: {epoch_test_acc/len(test_loader):.3f}')
    loss_list.append(epoch_loss/len(train_loader))
    accuracy.append(epoch_acc/len(train_loader))
    loss_list_test.append(epoch_test_loss/len(test_loader))
    accuracy_test.append(epoch_test_acc/len(test_loader))
    
    
import matplotlib.pyplot as plt
figure, axis = plt.subplots(1, 4, figsize=(24,24))
#X = np.arange(1, EPOCHS+1, 1)

##TRAIN
#loss
axis[0].plot(loss_list)
axis[0].set_title("Loss Function on train")

# accuracy
axis[1].plot(accuracy)
axis[1].set_title("F1 on train")

##TEST
#loss
axis[2].plot(loss_list_test)
axis[2].set_title("Loss Function on test")

# accuracy
axis[3].plot(accuracy_test)
axis[3].set_title("F1 on test")

plt.savefig("/home/preeti/Transformers/ebm/loss_and_accuracy_limited_branch_train_test.png")
plt.show()

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch,_ in train_loader:
        X_batch = torch.Tensor(X_batch)
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        #y_pred_list.append(i.item() for i in y_pred_tag)

y_pred = []
for i in y_pred_list:
    for j in i:
        y_pred.append(int(j))

print(classification_report(y_train, y_pred))
confusion_matrix(y_train, y_pred)

#torch.save(model.state_dict(), '/home/preeti/Transformers/ebm/final_upto_two_blk_50_new_loss.pt')



#on unseen data
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

#y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_pred = []
for i in y_pred_list:
    for j in i:
        y_pred.append(int(j))

print(classification_report(acc_test, y_pred))
confusion_matrix(acc_test, y_pred)





