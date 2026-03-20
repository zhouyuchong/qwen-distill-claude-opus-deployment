"""
Anthropic Messages API  →  OpenAI Chat API  轻量代理

将 Claude Code 发出的 Anthropic 格式请求翻译为 OpenAI 格式，
转发给 vLLM 后端，再将响应翻译回 Anthropic 格式。

支持：文本对话、工具调用(tool_use/tool_result)、流式输出

用法:
    python anthropic_proxy.py --vllm-url http://localhost:8766 --port 8800

然后:
    ANTHROPIC_BASE_URL=http://<host>:8800 ANTHROPIC_API_KEY=sk-placeholder claude
"""

import argparse
import json
import time
import uuid
import logging

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("anthropic-proxy")

app = FastAPI(
    title="Anthropic→OpenAI Proxy",
    max_request_body_size=134217728,  # 128MB, 默认值仅 10MB (10485760)
)

VLLM_URL: str = "http://localhost:8766"
SERVED_MODEL: str = "Qwen3.5-27B-Opus"
MAX_OUTPUT_TOKENS: int = 16384
HTTP_CLIENT: httpx.AsyncClient = None  # type: ignore


# ── request translation ─────────────────────────────────────────────

def _convert_content_blocks(content) -> tuple[str | None, list | None]:
    """Parse Anthropic content blocks into OpenAI text + tool_calls."""
    if isinstance(content, str):
        return content, None

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block["text"])
        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                text_parts.append(f"<thinking>\n{thinking}\n</thinking>")
        elif btype == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    text = "\n".join(text_parts) if text_parts else None
    return text, tool_calls or None


def anthropic_to_openai(body: dict) -> dict:
    """Translate a full Anthropic /v1/messages request to OpenAI /v1/chat/completions."""
    messages: list[dict] = []

    # system
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            parts = [b.get("text", "") for b in system if b.get("type") == "text"]
            if parts:
                messages.append({"role": "system", "content": "\n".join(parts)})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        # collect tool_result blocks separately – they become role=tool messages
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        other_blocks = [b for b in content if b.get("type") != "tool_result"]

        if other_blocks:
            text, tool_calls = _convert_content_blocks(other_blocks)
            m: dict = {"role": role}
            if tool_calls:
                m["content"] = text
                m["tool_calls"] = tool_calls
            else:
                m["content"] = text or ""
            messages.append(m)

        for tr in tool_results:
            rc = tr.get("content", "")
            if isinstance(rc, list):
                rc = "\n".join(
                    b.get("text", str(b)) for b in rc
                    if isinstance(b, dict)
                )
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": str(rc),
            })

    requested_max = body.get("max_tokens", 4096)
    capped_max = min(requested_max, MAX_OUTPUT_TOKENS)
    if requested_max != capped_max:
        log.info("Capping max_tokens: %d → %d", requested_max, capped_max)

    result = {
        "model": SERVED_MODEL,
        "messages": messages,
        "max_tokens": capped_max,
        "stream": body.get("stream", False),
    }

    for key in ("temperature", "top_p"):
        if key in body:
            result[key] = body[key]
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    if "tools" in body:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in body["tools"]
        ]

    return result


# ── response translation ────────────────────────────────────────────

_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


def _make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def openai_to_anthropic(oai: dict, model: str) -> dict:
    """Translate OpenAI chat completion response → Anthropic message."""
    choice = oai["choices"][0]
    message = choice["message"]

    content: list[dict] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": tc["function"]["name"],
            "input": args,
        })
    if not content:
        content.append({"type": "text", "text": ""})

    usage = oai.get("usage", {})
    return {
        "id": _make_msg_id(),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": _STOP_MAP.get(choice.get("finish_reason", "stop"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── streaming translation ───────────────────────────────────────────

async def _stream_anthropic(openai_req: dict, model: str):
    """Convert OpenAI SSE stream → Anthropic SSE stream."""
    msg_id = _make_msg_id()

    # message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    text_block_open = False
    text_block_idx = 0
    tool_blocks: dict[int, dict] = {}  # openai tool index → state
    next_block_idx = 0
    output_tokens = 0
    input_tokens = 0

    try:
        async with HTTP_CLIENT.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json=openai_req,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                yield _sse("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_body.decode()},
                })
                return

            buf = ""
            async for raw in resp.aiter_text():
                buf += raw
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    finish = choice.get("finish_reason")
                    usage = chunk.get("usage") or {}
                    if usage.get("prompt_tokens"):
                        input_tokens = usage["prompt_tokens"]

                    # text content
                    text = delta.get("content")
                    if text:
                        if not text_block_open:
                            text_block_idx = next_block_idx
                            next_block_idx += 1
                            yield _sse("content_block_start", {
                                "type": "content_block_start",
                                "index": text_block_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                            text_block_open = True
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": text_block_idx,
                            "delta": {"type": "text_delta", "text": text},
                        })
                        output_tokens += 1

                    # tool calls
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in tool_blocks:
                            if text_block_open:
                                yield _sse("content_block_stop", {
                                    "type": "content_block_stop", "index": text_block_idx,
                                })
                                text_block_open = False
                            bidx = next_block_idx
                            next_block_idx += 1
                            tool_blocks[idx] = {
                                "block_idx": bidx,
                                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": tc.get("function", {}).get("name", ""),
                            }
                            yield _sse("content_block_start", {
                                "type": "content_block_start",
                                "index": bidx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_blocks[idx]["id"],
                                    "name": tool_blocks[idx]["name"],
                                    "input": {},
                                },
                            })
                        args_delta = tc.get("function", {}).get("arguments", "")
                        if args_delta:
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": tool_blocks[idx]["block_idx"],
                                "delta": {"type": "input_json_delta", "partial_json": args_delta},
                            })

                    if finish:
                        if text_block_open:
                            yield _sse("content_block_stop", {
                                "type": "content_block_stop", "index": text_block_idx,
                            })
                        for tb in tool_blocks.values():
                            yield _sse("content_block_stop", {
                                "type": "content_block_stop", "index": tb["block_idx"],
                            })
                        yield _sse("message_delta", {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": _STOP_MAP.get(finish, "end_turn"),
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": output_tokens},
                        })
                        yield _sse("message_stop", {"type": "message_stop"})
                        return

    except Exception as e:
        log.exception("Streaming error")
        yield _sse("error", {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })

    # if we never got a finish_reason, close cleanly
    if text_block_open:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx})
    for tb in tool_blocks.values():
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": tb["block_idx"]})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── endpoints ────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def handle_messages(request: Request):
    body = await request.json()
    model = body.get("model", SERVED_MODEL)
    log.info("POST /v1/messages  model=%s stream=%s msgs=%d",
             model, body.get("stream"), len(body.get("messages", [])))

    openai_req = anthropic_to_openai(body)

    if body.get("stream", False):
        openai_req["stream"] = True
        openai_req["stream_options"] = {"include_usage": True}
        return StreamingResponse(
            _stream_anthropic(openai_req, model),
            media_type="text/event-stream",
        )

    resp = await HTTP_CLIENT.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=openai_req,
    )
    if resp.status_code != 200:
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": resp.text}},
            status_code=resp.status_code,
        )
    return JSONResponse(openai_to_anthropic(resp.json(), model))


@app.get("/v1/models")
async def handle_models():
    resp = await HTTP_CLIENT.get(f"{VLLM_URL}/v1/models")
    return JSONResponse(resp.json())


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path: str):
    """Forward unrecognized requests to vLLM."""
    body = await request.body()
    resp = await HTTP_CLIENT.request(
        request.method,
        f"{VLLM_URL}/{path}",
        content=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


# ── main ─────────────────────────────────────────────────────────────

def main():
    global VLLM_URL, SERVED_MODEL, MAX_OUTPUT_TOKENS, HTTP_CLIENT

    parser = argparse.ArgumentParser(
        description="Anthropic Messages API → OpenAI Chat API proxy for vLLM")
    parser.add_argument("--vllm-url", default="http://localhost:8766",
                        help="vLLM server base URL (default: http://localhost:8766)")
    parser.add_argument("--port", type=int, default=8800,
                        help="Proxy listen port (default: 8800)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-name", default="Qwen3.5-27B-Opus",
                        help="Model name to use in OpenAI requests")
    parser.add_argument("--max-output-tokens", type=int, default=16384,
                        help="Cap max_tokens per request (default: 16384)")
    args = parser.parse_args()

    VLLM_URL = args.vllm_url.rstrip("/")
    SERVED_MODEL = args.model_name
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    log.info("Proxy: %s:%d → vLLM: %s  model: %s", args.host, args.port, VLLM_URL, SERVED_MODEL)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
