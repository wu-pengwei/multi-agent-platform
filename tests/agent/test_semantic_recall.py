from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore


def keyword_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lower = text.lower()
        vectors.append([
            float("python" in lower or "代码" in lower),
            float("旅行" in lower or "北京" in lower),
            float("咖啡" in lower),
        ])
    return vectors


def test_append_history_indexes_and_recalls(tmp_path: Path):
    store = MemoryStore(tmp_path)
    store.init_semantic(keyword_embed)
    store.append_history("用户偏好使用 Python 编写数据处理代码。")
    store.append_history("用户周末喜欢喝咖啡。")
    store.set_last_dream_cursor(2)

    context = store.get_semantic_context_sync("请回忆我的 Python 代码偏好", top_k=2)

    assert "## Semantic Memories" in context
    assert "Python" in context


def test_context_builder_injects_semantic_memories(tmp_path: Path):
    builder = ContextBuilder(tmp_path)
    builder.memory.init_semantic(keyword_embed)
    builder.memory.append_history("用户希望所有示例都使用 Python。")
    builder.memory.set_last_dream_cursor(1)

    messages = builder.build_messages([], "请给我一段 Python 示例", channel="test", chat_id="1")

    assert "## Semantic Memories" in messages[0]["content"]
    assert "用户希望所有示例" in messages[0]["content"]


def test_recent_history_is_not_duplicated_in_semantic_context(tmp_path: Path):
    store = MemoryStore(tmp_path)
    store.init_semantic(keyword_embed)
    store.append_history("用户的 Python 代码偏好")
    assert store.get_semantic_context_sync("Python 代码", top_k=3) == ""


def test_semantic_recall_degrades_without_store(tmp_path: Path):
    builder = ContextBuilder(tmp_path)
    messages = builder.build_messages([], "普通问题")
    assert "## Semantic Memories" not in messages[0]["content"]


def test_async_embedder_is_skipped_on_sync_path(tmp_path: Path):
    async def async_embed(texts: list[str]):
        return keyword_embed(texts)

    store = MemoryStore(tmp_path)
    store.init_semantic(async_embed)
    store.append_history("Python 记忆")
    assert store.get_semantic_context_sync("Python") == ""
