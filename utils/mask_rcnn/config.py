import torch
# from torchvision.models.detection.anchor_utils import AnchorGenerator


class Config:

    def print(self):
        """
        打印当前配置参数
        """
        print("now Config is:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def __init__(self):
        self.DATASET_PATH = "dataset"
        # 存label的json,来源于annotations.json的categories有信息
        self.LABEL_JSON_PATH = "datasets/label.json"
        # 分类数量+1(背景)
        self.NUM_CLASSES = 4
        # 创建采样器,是否需要将相似宽高比的图像分配到一组组成batch
        self.ASPECT_RATIO_GROUP_FACTOR = 0
        self.BATCH_SIZE = 2
        # 训练的数据集
        self.TRAIN_IMG_FOLDER = 'datasets/images/train'
        self.TRAIN_ANN_FILE = 'datasets/COCO/train.json'

        # 验证的数据集
        self.VAL_IMG_FOLDER = 'datasets/images/val'
        self.VAL_ANN_FILE = 'datasets/COCO/val.json'
        # 定义设备
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 优化器参数
        self.LR = 0.0005
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0001
        # lr调度器参数
        self.LR_SCHEDULER_STEP_SIZE = 10
        self.LR_SCHEDULER_GAMMA = 0.1

        # 训练参数
        # 训练轮数
        self.EPOCHS_NUM = 200
        # 骨干网络上训练(不冻结)的层
        self.TRAINABLE_BACKBONE_LAYERS = 5
        # rpn的anchor设定
        anchor_sizes = ((32, 64), (64, 128), (128, 256), (256, 512), (512, 1024))
        # anchor_sizes = ((32, 64), (64, 128))
        aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
        # self.RPN_ANCHOR_GENERATOR = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.RPN_ANCHOR_GENERATOR = None
        # 注意,一旦使用RESUME_PTH,则此类的所有配置,注意是所有,都将会被所指的PTH内的config所覆盖
        # self.RESUME_PTH = "weights/maskrcnn/model_0.pth"
        self.RESUME_PTH = None

        # 输出目录
        self.OUT_DIR = "weights/maskrcnn"

        # 打印频率,多少的batch打印一次
        self.PRINT_FREQ = 5
