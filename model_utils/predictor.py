import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG
from tracker import Tracker
from model_utils import check_imgsz, non_max_suppression, scale_boxes, Results


class BasePredictor:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.results = None
        self.args = get_cfg(cfg, overrides)
        self.args.conf = 0.25  # default conf=0.25
        self.model = None
        self.data = self.args.data
        self.imgsz = check_imgsz(self.args.imgsz, min_dim=2)
        self.device = torch.device(overrides.get('device'))
        self.tracker = Tracker()

    def preprocess(self, im):
        shape = im.shape[:2]

        r = min(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1])
        new_unpad = (int(shape[1] * r + 0.5), int(shape[0] * r + 0.5))
        dw, dh = (self.imgsz[1] - new_unpad[0]) / 2, (self.imgsz[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(
            im, int(dh), int(dh + 0.5), int(dw), int(dw + 0.5), cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        im = np.ascontiguousarray(im[..., ::-1].transpose((2, 0, 1)))
        im = torch.from_numpy(im).to(self.device, non_blocking=True).unsqueeze(0)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        return im

    def postprocess(self, pred, img, orig_img):
        pred = non_max_suppression(
            prediction=pred,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )[0]

        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return [Results(orig_img, names=self.model.names, boxes=pred)]

    def __call__(self, source=None, model=None, *args, **kwargs):
        return list(self.stream_inference(source, *args, **kwargs))

    @torch.inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        im = self.preprocess(source)
        preds = self.model(im, augment=False, *args, **kwargs)
        self.results = self.postprocess(preds, im, source)
        self.tracker.update_tracker(self.results)
        yield from self.results

    def setup_model(self, model, verbose=True):
        self.model = AutoBackend(
            weights=model,
            device=torch.device(self.args.device),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )
        self.model.names = {
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
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))


class YOLO:
    def __init__(self, model: str) -> None:
        super().__init__()
        self.predictor = None  # reuse predictor
        self.model = model  # model object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object

    def track(self, source, **kwargs) -> list:
        if self.predictor is None:
            kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack method needs low confidence predictions as input
            kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
            kwargs["mode"] = "track"
            self.predictor = BasePredictor(overrides=kwargs)
            self.predictor.setup_model(model=self.model)
        return self.predictor(source=source)

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
