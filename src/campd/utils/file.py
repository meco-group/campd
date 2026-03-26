import yaml
from pydantic import BaseModel


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def load_params_from_yaml(path: str):
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_params_to_yaml(path: str, params: object) -> None:
    """Save a (YAML-serializable) object to disk.

    Uses safe_dump and preserves key order for readability.
    """
    with open(path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)


def to_yamlable(obj: object) -> object:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: to_yamlable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_yamlable(v) for v in obj]
    return obj
