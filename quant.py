import torch.nn as nn
import torch.nn.functional as F

# some base_op, such as ``Add``、``Concat``
from micronet.base_module.op import *

import micronet.compression.quantization.wqaq.dorefa.quantize as quant_dorefa
import micronet.compression.quantization.wqaq.iao.quantize as quant_iao

from model import efficientnetv2_s as create_model


"""
--w_bits --a_bits, 权重W和特征A量化位数
--q_type, 量化类型(0-对称, 1-非对称)
--q_level, 权重量化级别(0-通道级, 1-层级)
--weight_observer, weight_observer选择(0-MinMaxObserver, 1-MovingAverageMinMaxObserver)
--bn_fuse, 量化中bn融合标志
--bn_fuse_calib, 量化中bn融合校准标志
--pretrained_model, 预训练浮点模型
--qaft, qaft标志
--ptq, ptq标志
--percentile, ptq校准的比例
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=2).to(device)
quant_lenet_dorefa = quant_dorefa.prepare(model, inplace=False, a_bits=8, w_bits=8)
quant_lenet_iao = quant_iao.prepare(
    model,
    inplace=False,
    a_bits=8,
    w_bits=8,
    q_type=0,
    q_level=0,
    weight_observer=0,
    bn_fuse=False,
    bn_fuse_calib=False,
    pretrained_model=False,
    qaft=False,
    ptq=False,
    percentile=0.9999,
)

# if ptq == False, do qat/qaft, need train
# if ptq == True, do ptq, don't need train
# you can refer to micronet/compression/quantization/wqaq/iao/main.py

print("***ori_model***\n", model)
print("\n***quant_model_dorefa***\n", quant_lenet_dorefa)
print("\n***quant_model_iao***\n", quant_lenet_iao)

print("\nquant_model is ready")
print("micronet is ready")