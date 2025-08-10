# https://docs.ultralytics.com/models/yolo11/#performance-metrics
# https://docs.ultralytics.com/modes/export/#export-formats



from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")
model.export(format="engine", half=True, nms=True, simplify=True, dynamic=True) 