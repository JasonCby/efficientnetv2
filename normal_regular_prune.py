import sys

from model import efficientnetv2_s

from  utils import read_split_data
from  my_dataset import MyDataSet

sys.path.append("../..")
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms



parser = argparse.ArgumentParser()
parser.add_argument("--data", action="store", default="./data", help="dataset path")
parser.add_argument("--cpu", action="store_true", help="disables CUDA training")
# percent(剪枝率)
parser.add_argument("--percent", type=float, default=0.5, help="nin:0.5")
# 正常|规整剪枝标志
parser.add_argument(
    "--normal_regular",
    type=int,
    default=1,
    help="--normal_regular_flag (default: normal)",
)
# model层数
parser.add_argument("--layers", type=int, default=41, help="layers (default: 9)")
# 稀疏训练后的model
parser.add_argument(
    "--model",
    default="weights/model-0.pth",
    type=str,
    metavar="PATH",
    help="path to raw trained model (default: none)",
)
# 剪枝后保存的model
parser.add_argument(
    "--save",
    default="weights/efficientnetV2-jianzhi.pth",
    type=str,
    metavar="PATH",
    help="path to save prune model (default: none)",
)
args = parser.parse_args()
base_number = args.normal_regular
layers = args.layers
print(args)

if base_number <= 0:
    print("\r\n!base_number is error!\r\n")
    base_number = 1

model = efficientnetv2_s( num_classes=2)
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        model.load_state_dict(torch.load(args.model))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print("旧模型: ", model)
total = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            total += m.weight.data.shape[0]

# 确定剪枝的全局阈值
bn = torch.zeros(total)
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0]
            bn[index : (index + size)] = m.weight.data.abs().clone()
            index += size
y, j = torch.sort(bn)
thre_index = int(total * args.percent)
if thre_index == total:
    thre_index = total - 1
thre_0 = y[thre_index]

# ********************************预剪枝*********************************
pruned = 0
cfg_0 = []
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre_0).float()
            remain_channels = torch.sum(mask)

            if remain_channels == 0:
                print("\r\n!please turn down the prune_ratio!\r\n")
                remain_channels = 1
                mask[int(torch.argmax(weight_copy))] = 1

            # ******************规整剪枝******************
            v = 0
            n = 1
            if remain_channels % base_number != 0:
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)

                    y, j = torch.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_0.append(mask.shape[0])
            cfg.append(int(remain_channels))
            cfg_mask.append(mask.clone())
            print(
                "layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}".format(
                    k,
                    mask.shape[0],
                    int(torch.sum(mask)),
                    (mask.shape[0] - torch.sum(mask)) / mask.shape[0],
                )
            )
pruned_ratio = float(pruned / total)
print("\r\n!预剪枝完成!")
print("total_pruned_ratio: ", pruned_ratio)

# ********************************预剪枝后model测试*********************************
def test():
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                    transforms.CenterCrop(img_size[num_model][1]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(
        './data')



    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])




    batch_size = 8

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,

                                             collate_fn=test_dataset.collate_fn)
    model.eval()
    correct = 0

    for data, target in test_loader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100.0 * float(correct) / len(test_loader.dataset)
    print("Accuracy: {:.2f}%\n".format(acc))
    return


print("************预剪枝模型测试************")
if not args.cpu:
    model.cuda()

print(cfg)
# ********************************剪枝*********************************
newmodel =efficientnetv2_s( 2,cfg)
torch.save({"cfg": cfg, "state_dict": newmodel.state_dict()}, args.save)

# ******************************剪枝后model测试*********************************
print("新模型: ", newmodel)
print("**********剪枝后新模型测试*********")
model = newmodel


# ******************************剪枝后model保存*********************************
print("**********剪枝后新模型保存*********")
torch.save({"cfg": cfg, "state_dict": newmodel.state_dict()}, args.save)
print("**********保存成功*********\r\n")

# *****************************剪枝前后model对比********************************
print("************旧模型结构************")
print(cfg_0)
print("************新模型结构************")
print(cfg, "\r\n")
