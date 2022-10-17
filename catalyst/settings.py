from catalyst.tools.frozen_class import FrozenClass


class Settings(FrozenClass):
    """Catalyst settings."""

    def __init__(  # noqa: D107
        self,
    ):

        # stages
        self.stage_train_prefix: str = "train"
        self.stage_valid_prefix: str = "valid"
        self.stage_infer_prefix: str = "infer"

        # loader
        self.loader_train_prefix: str = "train"
        self.loader_valid_prefix: str = "valid"
        self.loader_infer_prefix: str = "infer"


DEFAULT_SETTINGS = Settings()
SETTINGS = DEFAULT_SETTINGS


__all__ = [
    "SETTINGS",
    "Settings",
]
