# flake8: noqa

from catalyst.contrib.nn.criterion.ce import (
    MaskCrossEntropyLoss,
    SymmetricCrossEntropyLoss,
)
from catalyst.contrib.nn.criterion.circle import CircleLoss
from catalyst.contrib.nn.criterion.contrastive import (
    ContrastiveDistanceLoss,
    ContrastiveEmbeddingLoss,
    ContrastivePairwiseEmbeddingLoss,
)
from catalyst.contrib.nn.criterion.gan import (
    GradientPenaltyLoss,
    MeanOutputLoss,
)
from catalyst.contrib.nn.criterion.huber import HuberLoss
from catalyst.contrib.nn.criterion.margin import MarginLoss
from catalyst.contrib.nn.criterion.triplet import (
    TripletLoss,
    TripletLossV2,
    TripletPairwiseEmbeddingLoss,
    TripletMarginLossWithSampler,
)
from catalyst.contrib.nn.criterion.wing import WingLoss
