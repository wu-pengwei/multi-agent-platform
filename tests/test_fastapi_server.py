import pytest

pytest.importorskip("fastapi")
from httpx import ASGITransport, AsyncClient

from nanobot.api.fastapi_server import create_fastapi_app


class FakeResponse:
    content = "收到请求"


class FakeAgent:
    async def process_direct(self, **kwargs):
        callback = kwargs.get("on_stream")
        if callback:
            await callback("收到")
            await callback("请求")
            end = kwargs.get("on_stream_end")
            if end:
                await end()
        return FakeResponse()


@pytest.mark.asyncio
async def test_fastapi_health_models_and_chat():
    app = create_fastapi_app(FakeAgent(), model_name="test-model")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == "test-model"
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "你好"}]},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "收到请求"


@pytest.mark.asyncio
async def test_fastapi_streaming_chat():
    app = create_fastapi_app(FakeAgent(), model_name="test-model")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "你好"}],
            },
        )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "收到" in response.text


@pytest.mark.asyncio
async def test_fastapi_rejects_multiple_messages():
    app = create_fastapi_app(FakeAgent())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
        )
    assert response.status_code == 400
