import torch
import numpy as np
from copy import deepcopy
from ultralytics.utils.plotting import Annotator, colors
from model_utils.utils import clip_boxes


class_names = {
    0: 'Big Bus',
    1: 'Heavy Truck',
    2: 'Medium Truck',
    3: 'Microbus',
    4: 'Minibus-Coaster',
    5: 'Motor Cycle',
    6: 'Pickup-4 Wheeler',
    7: 'Private Car-Sedan Car',
    8: 'Small Truck',
    9: 'Trailer Truck'
}

class Boxes:
    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        self.data = boxes
        self.orig_shape = orig_shape
        self.is_track = n == 7

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, -3] if self.is_track else None

    @property
    def xywh(self):
        return self.xyxy2xywh(self.xyxy)

    def xyxy2xywh(self, x):
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y


class Results:
    def __init__(self, orig_img, names, boxes=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.names = names
        self._keys = ["boxes"]

    def __getitem__(self, idx):
        return self._apply("__getitem__", idx)

    def update(self, boxes=None):
        self.boxes = Boxes(clip_boxes(boxes, self.orig_shape), self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        return self._apply("cpu")

    def numpy(self):
        return self._apply("numpy")

    def new(self):
        return Results(orig_img=self.orig_img, names=self.names)

    def plot(
        self, conf=True, line_width=None, font_size=None, font="Arial.ttf",
            img=None, labels=True, boxes=True, color_mode="class",
    ):
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        annotator = Annotator(
            im=deepcopy(self.orig_img),
            line_width=line_width,
            font_size=font_size,
            font=font,
            example=names,
        )

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, conf, _id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if _id is None else f"{_id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(i if color_mode == "instance" else c, True), rotated=False)
                annotator.circle_label(box, '', color=colors(i if color_mode == "instance" else c, True))

        return annotator.result()
