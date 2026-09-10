"""Heartbeat service for periodic agent wake-ups.
Heartbeat 是一个定时唤醒的服务，定期读取 HEARTBEAT.md 并用 LLM 决定是否有需要执行的“任务”，若需要则通过回调执行并可将结果通知出去
被动自检 — 周期性唤醒代理，检查是否有需要执行的任务（如定时事件、系统监控等），并通过回调机制触发相应的处理逻辑。
Heartbeat 适合“是否执行”需要语义判断的场景；若只是严格按时间执行、且需审计可靠性，优先用 cron。"""

from nanobot.heartbeat.service import HeartbeatService

__all__ = ["HeartbeatService"]
