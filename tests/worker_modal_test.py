from fastapi.testclient import TestClient

from workers import worker_modal


def test_normalize_modal_response_variants():
    assert worker_modal.normalize_modal_response("plain text") == {"response": "plain text"}
    assert worker_modal.normalize_modal_response({"response": "ok"}) == {"response": "ok"}
    assert worker_modal.normalize_modal_response(
        {"data": {"answer": "nested", "modal_task_id": "task-1"}}
    ) == {"response": "nested", "modal_task_id": "task-1"}


def test_generate_uses_prompt(monkeypatch):
    async def fake_call(prompt: str, max_new_tokens: int) -> dict:
        assert prompt == "hello"
        assert max_new_tokens == 32
        return {"response": "modal answer", "modal_task_id": "task-123"}

    monkeypatch.setattr(worker_modal, "call_modal_endpoint", fake_call)
    worker_modal.active_requests = 0
    worker_modal.total_requests = 0
    worker_modal.total_errors = 0

    with TestClient(worker_modal.worker_app) as client:
        response = client.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 32},
        )

    assert response.status_code == 200
    assert response.json() == {"response": "modal answer", "modal_task_id": "task-123"}
    assert worker_modal.active_requests == 0
    assert worker_modal.total_requests == 1
    assert worker_modal.total_errors == 0


def test_generate_accepts_question_alias(monkeypatch):
    async def fake_call(prompt: str, max_new_tokens: int) -> dict:
        assert prompt == "alias input"
        assert max_new_tokens == 12
        return {"response": "alias ok", "modal_task_id": "task-456"}

    monkeypatch.setattr(worker_modal, "call_modal_endpoint", fake_call)

    with TestClient(worker_modal.worker_app) as client:
        response = client.post(
            "/generate",
            json={"question": "alias input", "max_new_tokens": 12},
        )

    assert response.status_code == 200
    assert response.json() == {"response": "alias ok", "modal_task_id": "task-456"}


def test_generate_rejects_empty_prompt():
    with TestClient(worker_modal.worker_app) as client:
        response = client.post("/generate", json={"prompt": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Prompt cannot be empty."


def test_call_modal_endpoint_retries_with_query_params_on_422(monkeypatch):
    original_url = worker_modal.MODAL_WORKER_URL
    worker_modal.MODAL_WORKER_URL = "https://example.modal.run"

    calls = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, params=None, headers=None):
            calls.append({"url": url, "json": json, "params": params, "headers": headers})
            if len(calls) == 1:
                return worker_modal.httpx.Response(
                    422,
                    request=worker_modal.httpx.Request("POST", url),
                )
            return worker_modal.httpx.Response(
                200,
                json={"response": "ok", "modal_task_id": "task-789"},
                request=worker_modal.httpx.Request("POST", url),
            )

    monkeypatch.setattr(worker_modal.httpx, "AsyncClient", FakeAsyncClient)

    import asyncio

    result = asyncio.run(worker_modal.call_modal_endpoint("hello", 64))

    assert result == {"response": "ok", "modal_task_id": "task-789"}
    assert calls[0]["json"] == {"prompt": "hello", "max_new_tokens": 64}
    assert calls[0]["params"] is None
    assert calls[1]["json"] is None
    assert calls[1]["params"] == {"prompt": "hello", "max_new_tokens": 64}

    worker_modal.MODAL_WORKER_URL = original_url
