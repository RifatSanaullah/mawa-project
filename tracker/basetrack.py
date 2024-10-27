import sys
from enum import Enum

class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0
    _max_id = sys.maxsize

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self):
        return self.frame_id

    @classmethod
    def next_id(cls):
        cls._count += 1
        if cls._count > cls._max_id - 1:
            cls.reset_id()
        return cls._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @classmethod
    def reset_id(cls):
        cls._count = 1
