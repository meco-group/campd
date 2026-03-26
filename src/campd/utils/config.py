from typing import Dict
from experiment_launcher import Sweep
from typing import List, Type, Optional, Any, get_origin, get_args
from pydantic import BaseModel
from campd.utils.registry import Spec
import copy


def propagate_attributes(model: BaseModel) -> BaseModel:
    """
    Recursively propagates attributes from parent to nested child BaseModels.
    If a child model has a field with the same name as a field in the parent,
    the parent's value overwrites the child's value.

    Args:
        model: The root Pydantic model to process.

    Returns:
        The modified model (in-place modification).
    """
    # Iterate over all fields in the current model
    for field_name in type(model).model_fields:
        field_value = getattr(model, field_name)
        # Check if the field is a BaseModel
        if isinstance(field_value, BaseModel):
            child = field_value
            # Check for shared attributes between parent and child
            for child_field in type(child).model_fields:
                # Skip if the child field is the same as the field name holding the child
                # This prevents basic cycles like parent.config.config = parent.config
                if child_field == field_name:
                    continue

                if hasattr(model, child_field):
                    parent_val = getattr(model, child_field)
                    # Avoid self-reference assignments if possible
                    if parent_val is child:
                        continue
                    setattr(child, child_field, parent_val)

            # Recurse into the child
            propagate_attributes(child)

        elif isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, BaseModel):
                    for child_field in type(item).model_fields:
                        # For lists, field_name is the list name.
                        if child_field == field_name:
                            continue

                        if hasattr(model, child_field):
                            parent_val = getattr(model, child_field)
                            if parent_val is item:
                                continue
                            setattr(item, child_field, parent_val)

                    propagate_attributes(item)

    return model


def propagate_parent_attributes(cls):
    """
    Class decorator that automatically propagates parent attributes to children
    after initialization.
    """
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        propagate_attributes(self)

    cls.__init__ = __init__

    # Also wrap model_validate if it exists (Pydantic V2)
    if hasattr(cls, 'model_validate'):
        original_validate = cls.model_validate

        @classmethod
        def model_validate(cls, *args, **kwargs):
            instance = original_validate(*args, **kwargs)
            propagate_attributes(instance)
            return instance

        cls.model_validate = model_validate

    return cls


def propagate_attributes_dict(config: dict, model_cls: Optional[Type[BaseModel]] = None) -> dict:
    """
    Recursively propagates attributes from parent to nested child dictionaries.
    Mirroring `propagate_attributes`, if a child "model" (represented as dict)
    has a field with the same name as a field in the parent, the parent's value
    overwrites the child's value.

    Args:
        config: The configuration dictionary.
        model_cls: Optional Pydantic model class to use as schema.
                   If provided, keys can be added to children even if missing,
                   based on the schema. If not provided, only existing keys
                   in the child dict are considered targets for propagation.

    Returns:
        A new dictionary with propagated attributes.
    """

    config = copy.deepcopy(config)

    # Helper resolve type (handles Optional, List, etc.)
    def _resolve_type(t):
        origin = get_origin(t)
        if origin is None:
            return t
        args = get_args(t)
        if not args:
            return t
        # Unwrap Union/Optional (Union[Type, NoneType]) by taking first non-None arg
        # This is a simplification but sufficient for typical Pydantic models
        if origin is type(None):  # Should not happen for origin
            return t

        # Determine if it's a List
        if origin is list or origin is List:
            return List, _resolve_type(args[0])

        # Handle Optional/Union - take the first class type
        for arg in args:
            if isinstance(arg, type) and not isinstance(None, arg):  # Check against NoneType
                return _resolve_type(arg)

        return t

    # Iterate over items in the config to find potential children to recurse into
    for field_name, field_value in config.items():
        child_cls = None
        is_list = False

        if model_cls and field_name in model_cls.model_fields:
            field_info = model_cls.model_fields[field_name]
            # Inspect type
            resolved = _resolve_type(field_info.annotation)

            # Check if it returned a tuple (List, InnerType)
            if isinstance(resolved, tuple) and resolved[0] is List:
                is_list = True
                child_cls = resolved[1]
            elif isinstance(resolved, type) and issubclass(resolved, BaseModel):
                child_cls = resolved

        if field_value is None:
            continue

        if isinstance(field_value, dict) and (not model_cls or (child_cls and issubclass(child_cls, BaseModel))):
            child_dict = field_value

            if child_cls and issubclass(child_cls, Spec):
                continue

            # Propagation logic
            # Determine target fields on the child
            target_fields = []
            if child_cls:
                target_fields = child_cls.model_fields.keys()
            else:
                target_fields = child_dict.keys()

            for child_field in target_fields:
                if child_field == field_name:
                    continue  # Skip self-ref equivalent logic

                # Check if parent has this value
                if child_field in config:
                    parent_val = config[child_field]
                    # Assign to child
                    child_dict[child_field] = parent_val

            # Recurse
            config[field_name] = propagate_attributes_dict(
                child_dict, model_cls=child_cls)

        elif isinstance(field_value, list):
            # If we know it's a list of models, or if no schema and list of dicts
            is_valid_list = False
            if model_cls:
                if is_list and child_cls and issubclass(child_cls, BaseModel):
                    is_valid_list = True
            else:
                # No schema, assume list of dicts might need propagation
                is_valid_list = True

            if is_valid_list:
                for i, item in enumerate(field_value):
                    if isinstance(item, dict):
                        # Propagation logic for item
                        target_fields = []
                        if child_cls:
                            target_fields = child_cls.model_fields.keys()
                        else:
                            target_fields = item.keys()

                        for child_field in target_fields:
                            if child_field == field_name:
                                continue

                            if child_field in config:
                                item[child_field] = config[child_field]

                        # Recurse
                        field_value[i] = propagate_attributes_dict(
                            item, model_cls=child_cls)

    return config


def process_imports(imports: List[str], base_path: Optional[str] = None):
    """
    Process a list of imports.
    Entries can be python modules (e.g. "my_package.module") or file paths (e.g. "./custom_stuff.py").

    Args:
        imports: List of import strings.
        base_path: Base path to resolve relative file paths from. Usually the directory of the config file.
    """
    import sys
    import os
    import importlib.util

    if imports is None:
        return

    # Add base_path to sys.path to allow importing modules relative to the config file
    if base_path:
        abs_base = os.path.abspath(base_path)
        if abs_base not in sys.path:
            sys.path.insert(0, abs_base)

    for imp in imports:
        # Check if it is a file or directory path
        candidate_path = None
        if base_path and not os.path.isabs(imp):
            possible_path = os.path.join(base_path, imp)
            if os.path.exists(possible_path):
                candidate_path = os.path.abspath(possible_path)

        # Also check relative to CWD or absolute
        if not candidate_path and os.path.exists(imp):
            candidate_path = os.path.abspath(imp)

        if candidate_path:
            # It is a file or directory
            full_path = candidate_path

            # Check if it is a directory
            if os.path.isdir(full_path):
                # Directory autoload
                dir_path = full_path
                # Add directory to sys.path
                if dir_path not in sys.path:
                    sys.path.insert(0, dir_path)

                # Iterate over .py files
                # Sort to ensure deterministic load order
                for filename in sorted(os.listdir(dir_path)):
                    if filename.endswith(".py") and filename != "__init__.py":
                        file_path = os.path.join(dir_path, filename)
                        module_name = os.path.splitext(filename)[0]

                        spec = importlib.util.spec_from_file_location(
                            module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                        else:
                            # Warn or raise? Warning seems appropriate for individual file failure in batch
                            print(
                                f"Warning: Could not load spec for file in autoload: {file_path}")

            else:

                if full_path.endswith(".py"):
                    module_name = os.path.splitext(
                        os.path.basename(full_path))[0]
                    module_dir = os.path.dirname(full_path)

                    # Add directory to sys.path if not present to allow relative imports inside that module
                    if module_dir not in sys.path:
                        sys.path.insert(0, module_dir)

                    # Import the module
                    spec = importlib.util.spec_from_file_location(
                        module_name, full_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                    else:
                        raise ImportError(
                            f"Could not load spec for file: {full_path}")
                else:
                    importlib.import_module(imp)

        else:
            # Not a local path, treat as module
            importlib.import_module(imp)


def merge_dict_b_in_a(a: Dict, b: Dict) -> Dict:
    """
    Deep merge dict2 into dict1.
    """
    for key, value in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(value, dict):
            merge_dict_b_in_a(a[key], value)
        else:
            a[key] = value
    return a


def convert_dict_to_sweep_dict(dict1: Dict) -> Dict:
    """
    Convert a dictionary to a sweep dictionary.
    """
    sweep_dict = {}
    for key, value in dict1.items():
        if isinstance(value, dict):
            sweep_dict[key] = convert_dict_to_sweep_dict(value)
        else:
            _value = value if isinstance(value, list) else [value]
            sweep_dict[key] = Sweep(
                values=_value, name=key)
    return sweep_dict
