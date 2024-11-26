from module import load_dataset, load_config


def main() -> None:
    args = load_config()
    load_dataset(
        args["model"]["name"],
        args["files"]["root"],
        args["model"]["batch_size"],
        args["model"]["num_workers"],
    )


if __name__ == "__main__":
    main()
