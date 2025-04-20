from ultralytics import YOLO
import os
from utils.yolo.attention import add_attention

root = os.getcwd()
## 配置step1路径
pretrained_model_path = os.path.join(root, "yolo11n.pt")
yaml_path = os.path.join(root, "data.yaml")

## 配置step2路径
step1_train_model_path = os.path.join(root, 'runs/detect/train3/weights/best.pt')
step2_constraint_train_model_path = os.path.join(root, "runs/detect/Constraint")

## 配置step3路径
pruning_rate = 0.8
step3_prune_before_model_path = os.path.join(step2_constraint_train_model_path, "weights/last.pt")
step3_prune_after_model_path = os.path.join(step2_constraint_train_model_path, "weights/prune.pt")

## 配置step4路径
step4_finetune_model_path = os.path.join(root, "runs/detect/finetune")

## 配置step5路径
step5_teacher_model_path = step1_train_model_path
step5_student_model_path = os.path.join(step4_finetune_model_path, 'weights/best.pt')
step5_output_model_path = os.path.join(root, "runs/detect/student")

def step1_train():
    model = YOLO(pretrained_model_path)
    model.train(data=yaml_path, device="0", imgsz=640, epochs=50, batch=2, workers=0, save_period=1)  # train the model


def step2_constraint_train():
    model = YOLO(step1_train_model_path)
    model.train(data=yaml_path, device="0", imgsz=640, epochs=50, batch=2, amp=False, workers=0, save_period=1,
                name=step2_constraint_train_model_path)  # train the model


def step3_pruning():
    # from utils.yolo.seg_pruning import do_pruning  use for seg
    from utils.yolo.det_pruning import do_pruning  # use for det
    do_pruning(step3_prune_before_model_path, step3_prune_after_model_path, pruning_rate)


def step4_finetune():
    model = YOLO(step3_prune_after_model_path)  # load a pretrained model (recommended for training)
    for param in model.parameters():
        param.requires_grad = True
    model.train(data=yaml_path, device="0", imgsz=640, epochs=200, batch=2, workers=0,
                name=step4_finetune_model_path)  # train the model


def step5_distillation():
    layers = ["6", "8", "13", "16", "19", "22"]
    model_t = YOLO(step5_teacher_model_path)  # the teacher model
    model_s = YOLO(step5_student_model_path)  # the student model
    model_s = add_attention(model_s)

    model_s.train(data=yaml_path, Distillation=model_t.model, loss_type='mgd', layers=layers, amp=False, imgsz=1280,
                  epochs=300,
                  batch=2, device=0, workers=0, lr0=0.001, name=step5_output_model_path)


if __name__ == '__main__':
    # step1_train()
    # step2_constraint_train()
    # step3_pruning()
    # step4_finetune()
    step5_distillation()
