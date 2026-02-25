from art import text2art
import sys
from .config import (
    get_server_name,
    get_server_author,
    get_server_description,
    get_server_version,
)


def print_server_banner():
    """
    打印 MCP banner，包括：
    - MCP 名称（大字）
    - 作者
    - 版本号
    - 描述
    """
    name = get_server_name()
    author = get_server_author()
    version = get_server_version()
    description = get_server_description()

    ascii_name = text2art(name, font="bubble")
    print(ascii_name, file=sys.stderr)
    print(f"Author: {author}", file=sys.stderr)
    print(f"Version: {version}", file=sys.stderr)
    print(f"{description}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
