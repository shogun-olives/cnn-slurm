from module import create_job, load_config


def main() -> None:
    jobs = load_config("./config/jobs.yaml")
    for _, job in jobs.items():
        create_job(**job)


if __name__ == "__main__":
    main()
