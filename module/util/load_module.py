import types
import difflib


def get_module(module: types.ModuleType, name: str) -> any:
    """
    Get a module from a module.

    Args:
        module: The module containing the dataset.
        name (str): The name of the dataset.

    Returns:
        any: The dataset.
    """
    # Remove spaces and convert to lowercase
    name = name.lower().strip().replace(" ", "")
    dataset_names = {
        name.lower(): name for name in dir(module) if not name.startswith("_")
    }

    # If dataset exists return it
    if name in dataset_names:
        return getattr(module, dataset_names[name])

    # Otherwise, raise an error with suggestions
    matches = difflib.get_close_matches(name, dataset_names.keys(), cutoff=0)
    raise ValueError(
        f"{name} not in {module.__name__}. Did you mean {[dataset_names[m] for m in matches]}?"
    )
