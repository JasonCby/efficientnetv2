import os
import json

import torch.nn as nn
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = r"0.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/model-16.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    print(total)

    # # 确定剪枝的全局阈值
    # bn = torch.zeros(total)
    # index = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.shape[0]
    #         bn[index:(index + size)] = m.weight.data.abs().clone()
    #         index += size
    # # 按照权值大小排序
    # y, i = torch.sort(bn)
    # thre_index = int(total * 0.5)
    # # 确定要剪枝的阈值
    # thre = y[thre_index]
    # # ********************************预剪枝*********************************#
    # pruned = 0
    # cfg = []
    # cfg_mask = []
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, nn.BatchNorm2d):
    #         weight_copy = m.weight.data.abs().clone()
    #         # 要保留的通道标记Mask图
    #         mask = weight_copy.gt(thre).float()
    #         # 剪枝掉的通道数个数
    #         pruned = pruned + mask.shape[0] - torch.sum(mask)
    #         m.weight.data.mul_(mask)
    #         m.bias.data.mul_(mask)
    #         cfg.append(int(torch.sum(mask)))
    #         cfg_mask.append(mask.clone())
    #         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
    #               format(k, mask.shape[0], int(torch.sum(mask))))
    #     elif isinstance(m, nn.AdaptiveAvgPool2d):
    #         cfg.append('A')
    #
    # pruned_ratio = pruned / total
    #
    # print('Pre-processing Successful!')
    #
    # layer_id_in_cfg = 0
    # newmodel = create_model(num_classes=2).to(device)
    # start_mask = torch.ones(3)
    # end_mask = cfg_mask[layer_id_in_cfg]
    # for m0 in model.modules():
    #     if isinstance(m0, nn.Conv2d):
    #         print(m0.weight.data.shape)
    #     elif isinstance(m0, nn.BatchNorm2d):
    #         print(m0.weight.data.shape)
    # for [m0, m1] in zip(model.modules(), newmodel.modules()):
    #     # 对BN层和ConV层都要剪枝
    #     print(type(m0))
    #     if isinstance(m0, nn.BatchNorm2d):
    #         # np.squeeze 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    #         # np.argwhere(a) 返回非0的数组元组的索引，其中a是要索引数组的条件。
    #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    #         # 如果维度是1，那么就新增一维，这是为了和BN层的weight的维度匹配
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1, (1,))
    #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
    #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
    #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
    #         m1.running_var = m0.running_var[idx1.tolist()].clone()
    #         layer_id_in_cfg += 1
    #         # 注意start_mask在end_mask的前一层，这个会在裁剪Conv2d的时候用到
    #         start_mask = end_mask.clone()
    #         if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
    #             end_mask = cfg_mask[layer_id_in_cfg]
    #     elif isinstance(m0, nn.Conv2d):
    #         print(m0.weight.data.shape)
    #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    #         print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
    #         if idx0.size == 1:
    #             idx0 = np.resize(idx0, (1,))
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1, (1,))
    #         # 注意卷积核Tensor维度为[n, c, w, h]，两个卷积层连接，下一层的输入维度n就等于当前层的c
    #         print(idx0.shape, idx1.shape)
    #         print(idx0.tolist())
    #         print(m0.weight.data.shape)
    #         if(m0.weight.data.shape[1] != 1):
    #             w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
    #             w1 = w1[idx1.tolist(), :, :, :].clone()
    #         else:
    #             w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
    #         m1.weight.data = w1.clone()
    #     elif isinstance(m0, nn.Linear):
    #         # 注意卷积核Tensor维度为[n, c, w, h]，两个卷积层连接，下一层的输入维度n'就等于当前层的c
    #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #         if idx0.size == 1:
    #             idx0 = np.resize(idx0, (1,))
    #         m1.weight.data = m0.weight.data[:, idx0].clone()
    #         m1.bias.data = m0.bias.data.clone()
    #
    # torch.save(model.state_dict(), "./weights/model-{}.pth".format('pruned'))
    # print(newmodel)
    # model = newmodel


    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
