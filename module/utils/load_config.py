import yaml


def load_config() -> dict:
    """
    Load the configuration data.

    Returns:
        dict: The configuration data.
    """
    # Get config data
    with open("./config/default.yaml", "r") as file:
        default = yaml.safe_load(file)

    with open("./config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return merge_dicts(default, config)


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries.
    If a key is in both dictionaries, the value from dict2 will be used.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: The merged dictionary.
    """
    temp = dict1.copy()

    for key in dict2.keys():
        if key in temp and isinstance(dict2[key], dict):
            temp[key] = merge_dicts(temp[key], dict2[key])
        else:
            temp[key] = dict2[key]

    return temp
