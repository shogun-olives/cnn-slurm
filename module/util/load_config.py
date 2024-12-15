import yaml


def load_config(fn: str = "./config/config.yaml") -> dict:
    """
    Load the configuration data.

    Args:
        fn (str, optional): The configuration file. Defaults to "./config/config.py".

    Returns:
        dict: The configuration data.
    """
    # Get config data
    with open(fn, "r") as file:
        config = yaml.safe_load(file)

    return fmt_dict(config)


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


def fmt_dict(dictionary: dict) -> dict:
    """
    Format a dictionary.

    Args:
        dictionary (dict): The dictionary to format.

    Returns:
        dict: The formatted dictionary.
    """
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            dictionary[key] = fmt_dict(dictionary[key])
        else:
            dictionary[key] = fmt_sci_num(dictionary[key])

    return dictionary


def fmt_sci_num(val: str) -> float | int | str:
    """
    Format a scientific number from a string to a numeric.

    Args:
        val (str): The scientific number.

    Returns:
        float | str: The formatted number.
    """
    if not isinstance(val, str):
        return val

    try:
        temp = float(val)
    except ValueError:
        return val

    if temp.is_integer():
        return int(temp)
    else:
        return temp
