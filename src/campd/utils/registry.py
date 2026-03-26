from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, TypeVar, Protocol, runtime_checkable, Mapping, Union, Callable, Type
from typing import Generic

from pydantic import BaseModel, Field, ConfigDict
from .file import to_yamlable


T = TypeVar("T")

REGISTRIES: dict[str, Registry[T]] = {}


class Registry(Generic[T]):
    def __init__(self, name: str, *, forbid_duplicates: bool = True):
        self._name = name
        self._map: dict[str, Type[T]] = {}
        self._forbid_duplicates = forbid_duplicates
        REGISTRIES[name] = self

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if self._forbid_duplicates and key in self._map:
                # check if the class is the same (by module and name)
                prev_cls = self._map[key]
                if (prev_cls.__module__, prev_cls.__qualname__) != (cls.__module__, cls.__qualname__):
                    raise KeyError(
                        f"Duplicate registration '{key}' in registry '{self._name}'. "
                        f"Existing: {prev_cls.__module__}.{prev_cls.__qualname__}, "
                        f"New: {cls.__module__}.{cls.__qualname__}"
                    )
            self._map[key] = cls
            return cls
        return deco

    def __getitem__(self, key: str) -> Type[T]:
        try:
            return self._map[key]
        except KeyError as e:
            available = ", ".join(sorted(self._map.keys()))
            raise KeyError(
                f"Unknown '{key}' in registry '{self._name}'. Available: [{available}]") from e

    def available(self) -> list[str]:
        return sorted(self._map.keys())


@runtime_checkable
class FromCfg(Protocol):
    @classmethod
    def from_config(cls, cfg: Any) -> Any: ...


def import_string(path: str) -> Any:
    """
    Import a dotted path to a symbol.
    Examples:
      - "torch.optim.Adam" -> <class Adam>
      - "my_pkg.my_mod:MyClass" -> <class MyClass>  (colon supported)
    """
    if ":" in path:
        module_path, attr = path.split(":", 1)
    elif "." in path:
        module_path, attr = path.rsplit(".", 1)
    else:
        module_path = path
        attr = None

    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(
            f"Could not resolve '{attr}' from module '{module_path}'") from e


def _build_value(v: Any, registry: Optional[Mapping[str, Any]] = None) -> Any:
    """
    Recursively build nested Specs (and nested containers that contain Specs).
    """
    if isinstance(v, Spec):
        # Prefer registry if given, otherwise import path
        return v.build_from(registry) if registry is not None else v.build()

    if isinstance(v, dict):
        return {k: _build_value(x, registry) for k, x in v.items()}

    if isinstance(v, (list, tuple)):
        built = [_build_value(x, registry) for x in v]
        return built if isinstance(v, list) else type(v)(built)

    return v


class Spec(BaseModel, Generic[T]):
    """
    Serializable description of something buildable.

    - cls: import path or registry key (e.g. "torch.optim.Adam")
    - init: kwargs passed to constructor (supports nested Spec)
    - config: if provided, build using impl.from_config(config) (supports dict or Pydantic BaseModel)
    - registry: if provided, build using registry[cls] (supports dict or Pydantic BaseModel)

    Notes:
      - This model is frozen/immutable.
      - We never mutate `config` or `init` during build; we merge extra_kwargs immutably.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    cls: str
    init: Dict[str, Any] = Field(default_factory=dict)
    config: Optional[Union[dict, BaseModel]] = None
    registry: Optional[str] = None

    def _merge_config(self, extra_kwargs: Dict[str, Any]) -> Union[dict, BaseModel]:
        cfg = self.config
        if cfg is None:
            raise RuntimeError("_merge_config called with config=None")

        if isinstance(cfg, dict):
            # Immutable merge
            cfg_copy = cfg.copy()
            cfg_copy.update(extra_kwargs)
            return cfg_copy

        # Pydantic model: immutable update
        return cfg.model_copy(update=extra_kwargs)

    def _build_with_impl(self, impl: Any, *, registry: Optional[Mapping[str, Any]], **extra_kwargs: Any) -> T:
        # config-mode: require FromCfg
        if self.config is not None:
            if not isinstance(impl, type):
                raise TypeError(
                    f"Spec '{self.cls}' resolved to non-type '{type(impl)}' in config-mode.")

            if not issubclass(impl, FromCfg):  # runtime_checkable Protocol
                raise TypeError(
                    f"Spec '{self.cls}' was given 'config' but the resolved type does not implement "
                    f"from_config(config)."
                )

            merged_cfg = self._merge_config(dict(extra_kwargs))
            return impl.from_config(merged_cfg)

        # init-mode: standard constructor kwargs
        kwargs = _build_value(self.init, registry=registry)
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"Spec.init must build to a dict of kwargs, got: {type(kwargs)}")

        kwargs.update(extra_kwargs)
        if not callable(impl):
            raise TypeError(
                f"Spec '{self.cls}' resolved to non-callable '{impl}'")

        return impl(**kwargs)

    def build(self, **extra_kwargs: Any) -> T:
        """
        Build using import path resolution (cls must be importable).
        """
        if self.registry is not None:
            return self.build_from(REGISTRIES[self.registry], **extra_kwargs)
        impl = import_string(self.cls)
        return self._build_with_impl(impl, registry=None, **extra_kwargs)

    def build_from(self, registry: Registry, **extra_kwargs: Any) -> T:
        """
        Build using a registry mapping. If the key is missing, fall back to import_string(cls).
        """
        try:
            impl = registry[self.cls]
        except KeyError:
            # Fallback to import_string
            try:
                impl = import_string(self.cls)
            except ImportError:
                # If import fails and it wasn't in registry, raise helpful error
                raise KeyError(
                    f"Unknown '{self.cls}' in registry '{registry._name}' and could not be imported. Available: [{registry._map.keys()}]")
        return self._build_with_impl(impl, registry=registry, **extra_kwargs)


def impl_path(obj: object) -> str:
    t = obj.__class__
    return f"{t.__module__}.{t.__qualname__}"


def registry_key_for(registry, obj: object) -> str | None:
    # Best-effort: registry internals are simple key->type mapping.
    mapping = getattr(registry, "_map", {})
    for k, t in mapping.items():
        try:
            if isinstance(obj, t):
                return k
        except Exception:
            continue
    return None


def spec_payload(
    *,
    cfg: BaseModel,
    obj: object | None = None,
    cls: str | None = None,
    registry=None,
    registry_name: str | None = None,
) -> dict:
    """Build a Spec-shaped payload for components."""

    if obj is None and cls is None:
        raise ValueError("spec_payload requires either `obj` or `cls`.")

    cls_key = registry_key_for(registry, obj) if (
        registry is not None and obj is not None) else None
    cls_value = cls_key if cls_key is not None else (
        cls if cls is not None else impl_path(obj))

    payload = {
        "cls": cls_value,
        "config": to_yamlable(cfg),
    }
    if cls_key is not None and registry_name is not None:
        payload["registry"] = registry_name

    # Validate shape once (and emit YAML-safe payload)
    return Spec.model_validate(payload).model_dump(mode="json")
