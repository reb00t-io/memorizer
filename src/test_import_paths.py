import importlib


def test_new_package_paths_are_importable() -> None:
    modules = [
        "src.model.context",
        "src.model.memory",
        "src.model.message",
        "src.model.model",
        "src.chat.completion",
        "src.chat.chat",
    ]

    for module_name in modules:
        importlib.import_module(module_name)
