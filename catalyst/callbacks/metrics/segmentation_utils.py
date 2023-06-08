import numpy as np


class SegmentationMeter:
    def __init__(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray):
        if len(tp) != len(fp) or len(fp) != len(fn) or len(fn) != len(tn):
            raise ValueError("All arrays must have the same length")
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def reset(self):
        self.tp.fill(0)
        self.fp.fill(0)
        self.fn.fill(0)
        self.tn.fill(0)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-12)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-12)

    def fbeta(self, beta: float):
        p = self.precision
        r = self.recall
        return (1 + beta**2) * p * r / (beta**2 * p + r + 1e-12)

    @classmethod
    def empty(cls, num_thresholds: int):
        return cls(
            tp=np.zeros(num_thresholds, dtype=np.float32),
            fp=np.zeros(num_thresholds, dtype=np.float32),
            fn=np.zeros(num_thresholds, dtype=np.float32),
            tn=np.zeros(num_thresholds, dtype=np.float32),
        )

    def __add__(self, other):
        return SegmentationMeter(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )
    
    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        return self