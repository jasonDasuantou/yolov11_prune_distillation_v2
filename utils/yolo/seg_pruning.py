from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect, C3k2
from torch.nn.modules.container import Sequential
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class PRUNE():
    def __init__(self) -> None:
        self.threshold = None

    def get_threshold(self, model, factor=0.8):
        ws = []
        bs = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                w = m.weight.abs().detach()
                b = m.bias.abs().detach()
                ws.append(w)
                bs.append(b)
                print(name, w.max().item(), w.min().item(), b.max().item(), b.min().item())
                print()
        # keep
        ws = torch.cat(ws)
        self.threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]

    def prune_conv(self, conv1: Conv, conv2: Conv):
        ## Normal Pruning
        gamma = conv1.bn.weight.data.detach()
        beta = conv1.bn.bias.data.detach()

        keep_idxs = []
        local_threshold = self.threshold
        while len(keep_idxs) < 8:  ## 若剩余卷积核<8, 则降低阈值重新筛选
            keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
            local_threshold = local_threshold * 0.5
        n = len(keep_idxs)
        # n = max(int(len(idxs) * 0.8), p)
        print(n / len(gamma) * 100)
        conv1.bn.weight.data = gamma[keep_idxs]
        conv1.bn.bias.data = beta[keep_idxs]
        conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
        conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
        conv1.bn.num_features = n
        conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
        conv1.conv.out_channels = n

        if isinstance(conv2, list) and len(conv2) > 3 and conv2[-1]._get_name() == "Proto":
            proto = conv2.pop()
            proto.cv1.conv.in_channels = n
            proto.cv1.conv.weight.data = proto.cv1.conv.weight.data[:, keep_idxs]
        if conv1.conv.bias is not None:
            conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

        ## Regular Pruning
        if not isinstance(conv2, list):
            conv2 = [conv2]
        for item in conv2:
            if item is None: continue
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            if isinstance(item, Sequential):
                conv1 = item[0]
                conv = item[1].conv
                conv1.conv.in_channels = n
                conv1.conv.out_channels = n
                conv1.conv.groups = n
                conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs, :]
                conv1.bn.bias.data = conv1.bn.bias.data[keep_idxs]
                conv1.bn.weight.data = conv1.bn.weight.data[keep_idxs]
                conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
                conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
                conv1.bn.num_features = n
            conv.in_channels = n
            conv.weight.data = conv.weight.data[:, keep_idxs]

    def prune(self, m1, m2):
        if isinstance(m1, C3k2):  # C3k2 as a top conv
            m1 = m1.cv2
        if isinstance(m1, Sequential):
            m1 = m1[1]
        if not isinstance(m2, list):  # m2 is just one module
            m2 = [m2]
        for i, item in enumerate(m2):
            if isinstance(item, C3k2) or isinstance(item, SPPF):
                m2[i] = item.cv1

        self.prune_conv(m1, m2)


def do_pruning(modelpath, savepath):
    pruning = PRUNE()

    ### 0. 加载模型
    yolo = YOLO(modelpath)  # build a new model from scratch
    pruning.get_threshold(yolo.model, 0.65)  # 这里的0.8为剪枝率。

    ### 1. 剪枝C3k2 中的Bottleneck
    for name, m in yolo.model.named_modules():
        if isinstance(m, Bottleneck):
            pruning.prune_conv(m.cv1, m.cv2)

    ### 2. 指定剪枝不同模块之间的卷积核
    seq = yolo.model.model
    for i in [3, 5, 7, 8]:
        pruning.prune(seq[i], seq[i + 1])

    ### 3. 对检测头进行剪枝
    # 在P3层: seq[15]之后的网络节点与其相连的有 seq[16]、detect.cv2[0] (box分支)、detect.cv3[0] (class分支)
    # 在P4层: seq[18]之后的网络节点与其相连的有 seq[19]、detect.cv2[1] 、detect.cv3[1]
    # 在P5层: seq[21]之后的网络节点与其相连的有 detect.cv2[2] 、detect.cv3[2]
    detect: Detect = seq[-1]
    proto = detect.proto
    last_inputs = [seq[16], seq[19], seq[22]]
    colasts = [seq[17], seq[20], None]
    for idx, (last_input, colast, cv2, cv3, cv4) in enumerate(zip(last_inputs, colasts, detect.cv2, detect.cv3, detect.cv4)):
        if idx == 0:
            pruning.prune(last_input, [colast, cv2[0], cv3[0], cv4[0], proto])
        else:
            pruning.prune(last_input, [colast, cv2[0], cv3[0], cv4[0]])
        pruning.prune(cv2[0], cv2[1])
        pruning.prune(cv2[1], cv2[2])
        pruning.prune(cv3[0], cv3[1])
        pruning.prune(cv3[1], cv3[2])
        pruning.prune(cv4[0], cv4[1])
        pruning.prune(cv4[1], cv4[2])

    ### 4. 模型梯度设置与保存
    for name, p in yolo.model.named_parameters():
        p.requires_grad = True

    yolo.val(data='data.yaml', batch=2, device=0, workers=0)
    torch.save(yolo.ckpt, savepath)



if __name__ == "__main__":
    modelpath = "runs/segment/Constraint/weights/best.pt"
    savepath = "runs/segment/Constraint/weights/last_prune.pt"
    do_pruning(modelpath, savepath)
