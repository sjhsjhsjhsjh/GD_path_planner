import csv
import os
import queue
import threading
from typing import Iterable, Optional


class AsyncStepLogger:
    def __init__(
        self,
        csv_path: str,
        header: Iterable,
        flush_every: int = 256,
        queue_size: int = 4096,
    ):
        self.csv_path = str(csv_path)
        self.header = list(header)
        self.flush_every = max(1, int(flush_every))
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_token = object()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._header_written = (
            os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        )

    def start(self):
        if self._thread is not None:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="async-step-logger",
            daemon=True,
        )
        self._thread.start()

    def submit(self, row):
        if self._closed:
            return
        self.start()
        self.queue.put(row)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.start()
        self.queue.put(self._stop_token)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self):
        pending = []
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(self.header)
                self._header_written = True
                f.flush()

            while True:
                item = self.queue.get()
                if item is self._stop_token:
                    if pending:
                        writer.writerows(pending)
                        pending.clear()
                    f.flush()
                    return

                pending.append(item)
                if len(pending) >= self.flush_every:
                    writer.writerows(pending)
                    pending.clear()
                    f.flush()
