from ultralytics import YOLO

model = YOLO("weights/bestv4.pt")
model.export(
    format="engine",
    dynamic=True,
    batch=1,
    workspace=4,
    half=True,
    data="coco.yaml",
)

model = YOLO("weights/bestv4.engine", task="detect")
