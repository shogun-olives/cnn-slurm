from module.job import create_job
from module.util import load_config


def main() -> None:
    jobs = load_config("./config/jobs.yaml")
    for _, job in jobs.items():
        create_job(**job)


if __name__ == "__main__":
    main()
