import json
from pathlib import Path
from typing import Dict, List, Union


def save_config(
    config: Union[Dict, List],
    path: Union[str, Path],
    data_format: str = None,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """
    Saves config to file. Path must be either YAML or JSON.
    Args:
        config (Union[Dict, List]): config to save
        path (Union[str, Path]): path to save
        data_format: ``yaml``, ``yml`` or ``json``.
        encoding: Encoding to write file. Default is ``utf-8``
        ensure_ascii: Used for JSON, if True non-ASCII
        characters are escaped in JSON strings.
        indent: Used for JSON
    """
    path = Path(path)

    if data_format is not None:
        suffix = data_format
    else:
        suffix = path.suffix

    if suffix not in [".json", ".yml", ".yaml"]:
        raise RuntimeError(f"Unknown file format '{suffix}'")

    with path.open(encoding=encoding, mode="w") as stream:
        if suffix == ".json":
            json.dump(config, stream, indent=indent, ensure_ascii=ensure_ascii)
        elif suffix in [".yml", ".yaml"]:
            import yaml

            yaml.dump(config, stream)
