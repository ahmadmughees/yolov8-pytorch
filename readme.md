# YOLOv8 Model
A minimal, single-file implementation of the [YOLOv8 model](https://github.com/ultralytics/ultralytics) with PyTorch as the only dependency.

## Inference
Run inference with official pre-trained or custom weights. Note that weights saved from the [official YOLOv8 model repository](https://github.com/ultralytics/ultralytics) contain configuration and hyperparameters, which require multiple dependencies to load. This repo simplifies the process.

### Usage
In your Python environment with the yolov8 package, export the state dictionary using:
```python
import torch
from ultralytics import YOLO

model = YOLO(r"yolov8n.pt")
torch.save(model.model.state_dict(), r"best_state_dict.pt")
```

### Running Inference
Run inference with python script using the following commands:
```Python
import torch
from yolo import DetectionModel

model = DetectionModel(scale="n", num_classes=80)
model.load_state_dict(torch.load(r"best_state_dict.pt"))
print(model(torch.ones(3,3,640,640)))
```
### Understanding YOLOv8 Implementation
This repo provides a simplified implementation of the YOLOv8 model, making it easier to understand and follow the model's stack trace. The [official YOLOv8 model repository](https://github.com/ultralytics/ultralytics) is becoming a generic object detection repository, and this repo removes unnecessary control paths specific to the v8 variant.

### TODO:
- [ ] add NMS to filter out overlapping bounding boxes

### Disclaimer and Responsible Use:
This repo is intended for educational and research  purposes only. You may consult [ultralytics](https://github.com/ultralytics/ultralytics) for any legal guidance. 

### References
- https://github.com/ultralytics/ultralytics
- The inspiration of this project is derived form https://github.com/karpathy/llm.c