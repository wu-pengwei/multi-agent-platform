"""Slash command routing and built-in handlers. 负责“内置命令”的解析与分发，例如 /help, /stop, /team 等。"""

from nanobot.command.builtin import register_builtin_commands
from nanobot.command.router import CommandContext, CommandRouter

__all__ = ["CommandContext", "CommandRouter", "register_builtin_commands"]
