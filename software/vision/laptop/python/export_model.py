from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")
model.export(format="engine", half=True, nms=True, simplify=True, dynamic=True) 