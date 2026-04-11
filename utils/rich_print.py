from rich.console import Console
from rich.text import Text
from datetime import datetime

LEVELS = ["LOG", "WARN", "ERROR", "DEBUG", "INFO"]
LEVEL_COLORS = {
    "LOG": "green",
    "WARN": "yellow",
    "ERROR": "red",
    "DEBUG": "cyan",
    "INFO": "blue",
    "SUCC": "green",
    "FAIL": "red",
}

MAX_PREFIX_LEN = max(len(f"[{l}]:") for l in LEVELS)


def log(console: Console, level: str, message: str):
    """
    使用 rich 库在控制台打印带有时间戳和颜色的日志消息。

    :param console: rich Console 对象
    :param level: 日志级别（LOG, WARN, ERROR, DEBUG, INFO）
    :param message: 要打印的日志消息
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{level}]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)  # 冒号后对齐间距

    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS.get(level, 'white')}")
    text.append(spaces)
    text.append(message)

    console.print(text)
