from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
root = os.getcwd()

def main():
    model_path = os.path.join(root, 'runs/detect/Constraint/weights/prune.pt')
    model = YOLO(model_path)  # load a custom model
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_parameters}")

    # Validate the model
    metrics = model.val(data='data.yaml', device="0", batch=1, workers=0)  # no arguments needed, dataset and settings remembered.
    # model.export(format="onnx")
    # metrics.box.map()  # map50-95(B)
    # metrics.box.map50()  # map50(B)
    # metrics.box.map75()  # map75(B)
    # metrics.box.maps()  # a list contains map50-95(B) of each category
    # metrics.seg.map()  # map50-95(M)
    # metrics.seg.map50()  # map50(M)
    # metrics.seg.map75()  # map75(M)
    # metrics.seg.maps()  # a list contains map50-95(M) of each category
    # metrics.info()


if __name__ == '__main__':
    main()
