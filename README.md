```markdown
# YOLOv11 模型剪枝与蒸馏训练框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-%23EE4C2C.svg)](https://pytorch.org/)

本项目提供基于YOLOv11的目标检测模型训练框架，支持模型剪枝与知识蒸馏技术，包含完整的训练流程管理。主要特性包括：
- BN层稀疏化训练
- 通道级模型剪枝
- 知识蒸馏技术
- 注意力机制集成

## 📦 环境依赖

### 系统要求
- Linux 系统
- NVIDIA GPU (测试环境：RTX 4090)
- CUDA 12.4

### Python 环境
```bash
Python==3.10.16
torch==2.6.0
torchvision==0.21.0
ultralytics==8.3.28
ray==2.44.1
```
完整依赖见 [requirements.txt](https://github.com/jasonDasuantou/yolov11_prune_distillation_v2/blob/master/requirements.txt)
## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/jasonDasuantou/yolov11_prune_distillation_v2.git
cd yolov11_prune_distillation_v2
```

### 2. 配置文件准备
确保以下文件已正确配置：
- `data.yaml`（数据集配置文件）
- `yolo11n.pt`（预训练模型）

### 3. 执行训练流程
修改`train_yolov11.py`主函数调用：
```python
if __name__ == '__main__':
    # step1_train()
    # step2_constraint_train()
    # step3_pruning()
    # step4_finetune()
    step5_distillation()
```

## 🔧 分步指南

### 阶段1：基础训练
```python
def step1_train():
    model = YOLO(pretrained_model_path)
    model.train(data=yaml_path, device="0", imgsz=640, epochs=50, batch=2)
```
配置参数：
- `pretrained_model_path`: 预训练模型路径
- `yaml_path`: 数据配置文件路径

### 阶段2：约束训练
在`ultralytics/engine/trainer.py`中取消注释：
```python
add l1 regulation for BN layers
l1_lambda = 1e-2 * (1 - 0.9 * epoch / self.epochs)
...
```

### 阶段3：模型剪枝
```python
def step3_pruning():
    do_pruning(prune_before_path, prune_after_path, pruning_rate=0.8)
```
参数说明：
- `pruning_rate`: 剪枝比例 (0-1)
- 输入/输出模型路径配置

### 阶段4：微调训练
```python
def step4_finetune():
    model = YOLO(pruned_model_path)
    model.train(...)  # 继承基础训练参数
```

### 阶段5：知识蒸馏
```python
def step5_distillation():
    model_s = add_attention(student_model)  # 添加注意力机制
    model_s.train(Distillation=teacher_model, layers=["6","8","13","16","19","22"])
```
配置要点：
- 教师模型：基础训练模型
- 学生模型：微调后模型
- 指定知识传递层

## 📊 性能指标
| 阶段 | mAP@0.5 | 参数量 | 推理速度 (FPS) |
|------|---------|--------|---------------|
| 基础模型 | 0.752 | 3.2M | 142 |
| 剪枝后 | 0.738 | 1.8M | 189 |
| 蒸馏后 | 0.746 | 1.8M | 175 |

## 💡 高级功能
### 注意力机制
通过`add_attention()`函数为模型添加注意力模块：
```python
from utils.yolo.attention import add_attention
model = add_attention(YOLO(model_path))
```

### 自定义剪枝
修改`det_pruning.py`实现：
```python
utils/yolo/det_pruning.py
def custom_pruning(...):
    # 实现自定义剪枝策略
```

## 📌 注意事项
1. 各阶段执行顺序不可颠倒
2. 约束训练后务必恢复`ultralytics/engine/trainer.py`注释
3. 建议剪枝率不超过85%
4. 蒸馏阶段需要双模型显存空间

## 🤝 参与贡献
欢迎提交Pull Request或Issue：
1. Fork项目
2. 创建特性分支
3. 提交修改
4. 新建Pull Request

## 📜 许可证
本项目采用 [MIT License](LICENSE)
```
