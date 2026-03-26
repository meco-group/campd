import pytest
from typing import List
from pydantic import BaseModel
from campd.utils.config import propagate_parent_attributes, propagate_attributes_dict


class SubConfig(BaseModel):
    name: str = "sub"
    val: int = 1
    shared: str = "child_shared"


@propagate_parent_attributes
class MainConfig(BaseModel):
    name: str = "main"
    shared: str = "parent_shared"
    sub: SubConfig


def test_propagate_parent_attributes_nested():
    sub = SubConfig()
    main = MainConfig(sub=sub)

    # Before propagation - NOW AFTER PROPAGATION because of init decorator
    # Note: With the decorator, propagation happens immediately on init.
    # So we can straight away assert equality.

    # After propagation, child should have parent's value
    assert main.sub.shared == "parent_shared"

    # After propagation, child should have parent's value
    assert main.sub.shared == "parent_shared"
    # Non-shared attributes should remain untouched
    assert main.sub.val == 1
    # Name is also shared (both have 'name'), so it should be overwritten
    assert main.sub.name == "main"


def test_deeply_nested():
    class DeepConfig(BaseModel):
        shared: str = "deep_child"

    class MidConfig(BaseModel):
        shared: str = "mid_child"
        deep: DeepConfig

    @propagate_parent_attributes
    class TopConfig(BaseModel):
        shared: str = "top"
        mid: MidConfig

    deep = DeepConfig()
    mid = MidConfig(deep=deep)
    top = TopConfig(mid=mid)

    # Top overrides mid
    assert top.mid.shared == "top"
    # Top overrides deep (propagated down)
    # Note: Logic propagates recursively.
    # 1. Top propagates "top" to Mid.shared.
    # 2. Then recurse into Mid.
    # 3. Mid now has shared="top". Mid propagates "top" to Deep.shared.
    assert top.mid.deep.shared == "top"


def test_propagate_attributes_dict_simple():
    """Test simple dictionary propagation without schema."""
    config = {
        "shared": "parent",
        "child": {
            "shared": "child",
            "val": 1
        }
    }

    config = propagate_attributes_dict(config)

    assert config["child"]["shared"] == "parent"
    assert config["child"]["val"] == 1


def test_propagate_attributes_dict_schema():
    """Test dictionary propagation using schema to add missing keys."""
    class ChildModel(BaseModel):
        shared: str = "child_default"
        other: int = 10

    class ParentModel(BaseModel):
        shared: str = "parent"
        child: ChildModel

    config = {
        "shared": "parent_val",
        "child": {
            "other": 99
            # "shared" is missing here
        }
    }

    # Without schema, "shared" would NOT be propagated because it's not in the child dict
    # With schema, it SHOULD be propagated
    config = propagate_attributes_dict(config, model_cls=ParentModel)

    assert "shared" in config["child"]
    assert config["child"]["shared"] == "parent_val"
    assert config["child"]["other"] == 99


def test_propagate_attributes_dict_list():
    """Test propagation into lists of dictionaries."""

    class ItemModel(BaseModel):
        shared: str = "item_default"
        id: int

    class ListModel(BaseModel):
        shared: str = "parent"
        items: List[ItemModel]

    config = {
        "shared": "parent_val",
        "items": [
            {"id": 1, "shared": "item1"},
            {"id": 2}  # "shared" missing
        ]
    }

    config = propagate_attributes_dict(config, model_cls=ListModel)

    assert config["items"][0]["shared"] == "parent_val"
    assert config["items"][1]["shared"] == "parent_val"


def test_propagate_attributes_standalone():
    """Test propagate_attributes function directly."""
    from campd.utils.config import propagate_attributes

    class Child(BaseModel):
        shared: str = "child"
        other: int = 1

    class Parent(BaseModel):
        shared: str = "parent"
        child: Child

    child = Child()
    parent = Parent(child=child)

    # Before propagation, child has default
    assert parent.child.shared == "child"

    propagate_attributes(parent)

    # After propagation, child has parent's value
    assert parent.child.shared == "parent"
    assert parent.child.other == 1


def test_propagate_attributes_list():
    """Test propagate_attributes with list of models."""
    from campd.utils.config import propagate_attributes

    class Item(BaseModel):
        shared: str = "item_default"
        id: int

    class Container(BaseModel):
        shared: str = "container"
        items: List[Item]

    items = [Item(id=1, shared="i1"), Item(id=2)]
    container = Container(items=items)

    propagate_attributes(container)

    assert container.items[0].shared == "container"
    assert container.items[1].shared == "container"


def test_propagate_attributes_dict_optional():
    """Test propagate_attributes_dict with Optional types."""
    from typing import Optional

    class Child(BaseModel):
        shared: str = "child"

    class Parent(BaseModel):
        shared: str = "parent"
        child: Optional[Child] = None

    # Case 1: Child is present
    config = {
        "shared": "parent_val",
        "child": {}
    }
    config = propagate_attributes_dict(config, model_cls=Parent)
    assert config["child"]["shared"] == "parent_val"


def test_propagate_attributes_dict_nested_mix():
    """Test propagate_attributes_dict with mixed list/dict nesting."""
    class Leaf(BaseModel):
        shared: str = "leaf"

    class Node(BaseModel):
        shared: str = "node"
        leaves: List[Leaf]

    class Root(BaseModel):
        shared: str = "root"
        node: Node

    config = {
        "shared": "root_val",
        "node": {
            "leaves": [{}, {}]
        }
    }

    config = propagate_attributes_dict(config, model_cls=Root)

    assert config["node"]["shared"] == "root_val"
    assert config["node"]["leaves"][0]["shared"] == "root_val"
    assert config["node"]["leaves"][1]["shared"] == "root_val"


def test_propagate_attributes_no_overwrite_different_names():
    """Test that unrelated fields are not overwritten."""
    from campd.utils.config import propagate_attributes

    class Child(BaseModel):
        unique_child: str = "unique"

    class Parent(BaseModel):
        shared: str = "parent"
        child: Child

    child = Child()
    parent = Parent(child=child)

    propagate_attributes(parent)

    assert parent.child.unique_child == "unique"
    # Ensure no new attribute 'shared' was added to child instance (Pydantic models are fixed schema)
    with pytest.raises(AttributeError):
        _ = parent.child.shared


def test_propagate_attributes_dict_adds_missing_keys_with_schema():
    """Test that missing keys in child dict are added if present in parent and schema."""
    class Child(BaseModel):
        attr_to_add: str = "default"
        existing_attr: int = 1

    class Parent(BaseModel):
        attr_to_add: str = "parent_val"
        child: Child

    config = {
        "attr_to_add": "parent_provided",
        "child": {
            "existing_attr": 2
            # "attr_to_add" is MISSING here
        }
    }

    # Before propagation, key is missing
    assert "attr_to_add" not in config["child"]

    config = propagate_attributes_dict(config, model_cls=Parent)

    # After propagation, key should be added from parent
    assert "attr_to_add" in config["child"]
    assert config["child"]["attr_to_add"] == "parent_provided"
    assert config["child"]["existing_attr"] == 2
