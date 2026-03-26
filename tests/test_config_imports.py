
import pytest
import os
import sys
import yaml
from campd.utils.config import process_imports
from campd.experiments.base import BaseExperiment, ExperimentCfg

# Helper to create a dummy python module


def create_dummy_module(path, content="foo = 'bar'"):
    with open(path, "w") as f:
        f.write(content)


def test_process_imports_module():
    # Test importing a standard library module (already imported, but function should handle it)
    process_imports(["os"])
    assert "os" in sys.modules


def test_process_imports_file(tmp_path):
    # Create a dummy python file
    module_path = tmp_path / "custom_module.py"
    create_dummy_module(module_path, "TEST_VAR = 123")

    # Process import
    process_imports([str(module_path.name)], base_path=str(tmp_path))

    # Check if imported
    assert "custom_module" in sys.modules
    assert sys.modules["custom_module"].TEST_VAR == 123

    # Clean up
    del sys.modules["custom_module"]


def test_process_imports_relative_path(tmp_path):
    # Create a dummy python file
    module_path = tmp_path / "subdir" / "other_module.py"
    os.makedirs(module_path.parent)
    create_dummy_module(module_path, "VAL = 456")

    # Process import relative to tmp_path
    process_imports(["subdir/other_module.py"], base_path=str(tmp_path))

    assert "other_module" in sys.modules
    assert sys.modules["other_module"].VAL == 456

    del sys.modules["other_module"]


def test_process_imports_not_found(tmp_path):
    with pytest.raises(ImportError):
        process_imports(["non_existent_file.py"], base_path=str(tmp_path))


def test_process_imports_module_in_config_dir(tmp_path):
    # Test importing a module by name (no .py) that is in the config directory
    # This requires config dir to be added to sys.path
    module_name = "local_module_imp"
    module_file = tmp_path / f"{module_name}.py"
    create_dummy_module(module_file, "LOCAL_VAR = 888")

    # Process import as module name
    process_imports([module_name], base_path=str(tmp_path))

    assert module_name in sys.modules
    assert sys.modules[module_name].LOCAL_VAR == 888

    del sys.modules[module_name]


def test_process_imports_parent_dir_relative(tmp_path):
    # Test importing a file from parent directory using .. syntax

    # Structure:
    # tmp_path/
    #   parent_mod.py
    #   subdir/
    #     config_dir/

    parent_mod_name = "parent_mod_test"
    parent_file = tmp_path / f"{parent_mod_name}.py"
    create_dummy_module(parent_file, "PARENT_VAR = 999")

    config_base_path = tmp_path / "subdir" / "config_dir"
    os.makedirs(config_base_path)

    # Import path: ../../parent_mod_test.py
    import_path = f"../../{parent_mod_name}.py"

    process_imports([import_path], base_path=str(config_base_path))

    assert parent_mod_name in sys.modules
    assert sys.modules[parent_mod_name].PARENT_VAR == 999

    del sys.modules[parent_mod_name]


def test_process_imports_directory(tmp_path):
    # Test autoloading a directory

    # Structure:
    # tmp_path/
    #   autoload_dir/
    #     mod1.py
    #     mod2.py
    #     __init__.py (should be ignored)
    #     not_py.txt

    autoload_dir = tmp_path / "autoload_dir"
    os.makedirs(autoload_dir)

    create_dummy_module(autoload_dir / "mod1.py", "VAR1 = 111")
    create_dummy_module(autoload_dir / "mod2.py", "VAR2 = 222")
    create_dummy_module(autoload_dir / "__init__.py", "INIT_VAR = 333")
    with open(autoload_dir / "not_py.txt", "w") as f:
        f.write("text")

    process_imports([str(autoload_dir.name)], base_path=str(tmp_path))

    assert "mod1" in sys.modules
    assert sys.modules["mod1"].VAR1 == 111
    assert "mod2" in sys.modules
    assert sys.modules["mod2"].VAR2 == 222

    # __init__.py shouldn't be loaded as a module named __init__ or init
    assert "__init__" not in sys.modules

    del sys.modules["mod1"]
    del sys.modules["mod2"]


class MockExperiment(BaseExperiment):
    CfgClass = ExperimentCfg
    def run(self): pass


def test_runner_loading_logic_with_imports(tmp_path):
    # Simulate runner logic
    module_name = "runner_test_mod"
    module_file = tmp_path / f"{module_name}.py"
    create_dummy_module(module_file, "RUNNER_VAL = 555")

    config_file = tmp_path / "runner_config.yaml"
    config_data = {
        "imports": [f"{module_name}.py"],
        "experiment": {
            "results_dir": str(tmp_path / "results_runner"),
            "seed": 888,
            "cls": "mock_cls"
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Runner steps:
    with open(config_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    imports = cfg_dict.pop("imports", None)
    process_imports(imports, base_path=str(tmp_path))

    if "experiment" in cfg_dict:
        cfg_dict = cfg_dict["experiment"]

    exp = MockExperiment(ExperimentCfg.model_validate(cfg_dict))

    assert exp.cfg.seed == 888
    assert module_name in sys.modules
    assert sys.modules[module_name].RUNNER_VAL == 555

    del sys.modules[module_name]


def test_base_experiment_flat_structure(tmp_path):
    # Old structure check: base.from_yaml should still work for simple configs
    config_file = tmp_path / "flat_config.yaml"
    config_data = {
        "results_dir": str(tmp_path / "results_flat"),
        "seed": 111,
        "cls": "mock_cls"
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    exp = MockExperiment(ExperimentCfg.model_validate(config_data))
    assert exp.cfg.seed == 111
