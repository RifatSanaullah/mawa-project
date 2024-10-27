import argparse
import torch
from tracker.byte_tracker import BYTETracker

__all__ = [
    "Tracker",
]

class Tracker:
    def __init__(self) -> None:
        cfg = argparse.Namespace()
        cfg.match_thresh = 0.8  # threshold for matching tracks
        cfg.new_track_thresh = 0.6  # threshold for init new track if the detection does not match any tracks
        cfg.track_buffer = 30  # buffer to calculate the time when to remove tracks
        cfg.track_high_thresh = 0.5  # threshold for the first association
        cfg.track_low_thresh = 0.1  # threshold for the second association
        cfg.fuse_score = True  # Whether to fuse confidence scores with the iou distances before matching

        tracker = BYTETracker(args=cfg, frame_rate=30)
        self.tracker = tracker

    def update_tracker(self, results) -> None:
        det = results[0].boxes.cpu().numpy()
        if len(det) == 0:
            return
        tracks = self.tracker.update(det)
        if len(tracks) == 0:
            return

        idx = tracks[:, -1].astype(int)
        results[0] = results[0][idx]

        update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
        results[0].update(**update_args)
