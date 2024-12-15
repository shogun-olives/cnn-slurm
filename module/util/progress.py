import time


class ProgressMessage:
    """
    Message for progress bar
    """

    def __init__(self, total: int, title: str = None, length: int = 50):
        self.pretext = f"{title:<18}  |" if title else "|"
        self.total = total
        self.start = time.time()
        self.length = length if length < 50 else 50 if length > 0 else 1

    def __call__(self, progress: int) -> str:
        percentage = float(progress) / self.total
        if progress == self.total:
            prog_bar = "#" * self.length
        else:
            prog_perc = percentage * self.length
            prog_bar = (
                "#" * int(prog_perc)
                + str(int(prog_perc * 10) % 10)
                + " " * (self.length - int(prog_perc) - 1)
            )
        t_delta = time.time() - self.start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(t_delta))
        return (
            self.pretext
            + prog_bar
            + f"|  {progress}/{self.total} [{elapsed}] {t_delta/max(progress, 1):.3f} s/it"
        )


class ProgressBar:
    """
    Progress bar for loops

    Example:
    ```python
    for i in Progress(100, title="counting"):
        sleep(0.1)

    array = ["hello", "world", "testing!"]
    for x in Progress(array, title="messages"):
        Progress.write(x)
    ```

    Args:
        iterator (int | range): The iterator to loop over.
        title (str): The title of the progress bar.
        length (int): The length of the progress bar.
        write_end (bool): Whether to write the end message or not
    """

    Existing = []

    def __init__(
        self, iterator, title: str = None, length: int = 50, write_end: bool = True
    ):
        if isinstance(iterator, int):
            iterator = range(iterator)
        self.iterator = iterator
        self.title = title
        self.write_end = write_end if len(ProgressBar.Existing) == 0 else False
        self.msg = None
        self.length = length

    def __iter__(self):
        message = ProgressMessage(len(self.iterator), self.title, self.length)

        if ProgressBar.Existing:
            print(ProgressBar.Existing[-1])

        ProgressBar.Existing.append(self)
        for i, x in enumerate(self.iterator):
            self.msg = message(i)
            print(self.msg, end="\r")
            yield x

        ProgressBar.Existing.pop()
        if ProgressBar.Existing:
            print("\033[K\033[F\033[K", end="")

        if self.write_end:
            print(message(len(self.iterator)))
        else:
            print("\033[K", end="")

    def __str__(self):
        return self.msg

    def write(*args, sep=" "):
        print("\r\033[K", end="")
        print(*args, sep=sep)


def main() -> None:
    # Test of progress bar
    num = 100
    for i in ProgressBar(10, "outter"):
        for _ in ProgressBar(15, str(i)):
            for _ in ProgressBar(900, "inner"):
                pass


if __name__ == "__main__":
    main()
