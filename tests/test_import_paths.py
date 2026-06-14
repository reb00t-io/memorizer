import importlib


def test_new_package_paths_are_importable() -> None:
    modules = [
        "memorizer",
        "memorizer.model.context",
        "memorizer.model.memory",
        "memorizer.model.message",
        "memorizer.model.model",
        "memorizer.chat.completion",
        "memorizer.chat.chat",
    ]

    for module_name in modules:
        importlib.import_module(module_name)
