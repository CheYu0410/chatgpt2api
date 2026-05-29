from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    collect_image_outputs,
    collect_text,
    count_message_tokens,
    count_text_tokens,
    encode_images,
    normalize_messages,
    stream_image_outputs_with_pool,
    stream_text_deltas,
    text_backend,
)
from utils.helper import build_chat_image_markdown_content, extract_chat_image, extract_chat_prompt, is_image_chat_request, parse_image_count


# ==================== Tool calling helpers ====================

def _compact_schema(schema: dict[str, Any] | None) -> str:
    """Compact JSON Schema to a short type signature string.
    e.g. {"type":"object","properties":{"file_path":{"type":"string"}}} -> {file_path!: string}
    """
    if not schema or not schema.get("properties"):
        return "{}"
    props = schema["properties"]
    required = set(schema.get("required") or [])
    parts = []
    for name, prop in props.items():
        if not isinstance(prop, dict):
            continue
        t = prop.get("type") or "any"
        if prop.get("enum"):
            t = "|".join(str(v) for v in prop["enum"])
        elif t == "array" and isinstance(prop.get("items"), dict):
            item_type = prop["items"].get("type") or "any"
            t = f"{item_type}[]"
        elif t == "object" and prop.get("properties"):
            t = _compact_schema(prop)
        req = "!" if name in required else "?"
        parts.append(f"{name}{req}: {t}")
    return "{" + ", ".join(parts) + "}"


def build_tool_instructions(tools: list[dict[str, Any]] | None, tool_choice: Any = None) -> str:
    """Build a system prompt fragment that instructs the model to output tool calls
    using a structured text format that is easy to parse with regex.
    Format: [TOOL: tool_name, param1: value1, param2: value2]
    """
    if not tools:
        return ""

    lines = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(tool.get("name") or fn.get("name") or "").strip()
        desc = str(tool.get("description") or fn.get("description") or "").strip()
        schema = tool.get("input_schema") or tool.get("parameters") or fn.get("input_schema") or fn.get("parameters")

        param_str = ""
        if schema and isinstance(schema, dict):
            props = schema.get("properties", {})
            if props:
                param_names = ", ".join(props.keys())
                param_str = f" (params: {param_names})"

        if name:
            lines.append(f"- {name}: {desc}{param_str}" if desc else f"- {name}{param_str}")

    if not lines:
        return ""

    # tool_choice constraint
    force_constraint = ""
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "any":
            force_constraint = '\n\n**IMPORTANT**: You MUST use at least one [TOOL: ...] call in your response. Do not respond with plain text only.'
        elif tool_choice.get("type") == "tool":
            required_name = tool_choice.get("name", "")
            force_constraint = f'\n\n**IMPORTANT**: You MUST call the "{required_name}" tool using the [TOOL: ...] format. Do not respond with plain text only.'

    return (
        "You have access to the following tools. When you need to use a tool, output it in this exact format (one tool per line):\n\n"
        "[TOOL: tool_name, parameter_name: parameter_value, ...]\n\n"
        "Rules:\n"
        "- Output ONLY the [TOOL: ...] line when calling a tool. No extra text, no markdown, no code blocks.\n"
        "- If multiple tools needed, output multiple [TOOL: ...] lines.\n"
        "- If no tool needed, respond normally in plain text.\n\n"
        "Available tools:\n"
        + "\n".join(lines)
        + "\n\n"
        + force_constraint
    )


def _tolerant_json_parse(text: str) -> dict[str, Any] | None:
    """Try to parse JSON, with fallback for truncated content.
    Attempts to close unclosed brackets/braces to recover partial tool calls.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to recover truncated JSON by closing open brackets
    # Count open braces/brackets and try appending closers
    text = text.strip()
    if not text:
        return None

    # Simple approach: try appending closing chars
    for suffix in ["", "}", "}]", "}]}", "}}", "}}}" ]:
        try:
            return json.loads(text + suffix)
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def parse_tool_calls(response_text: str) -> tuple[list[dict[str, Any]], str]:
    """Parse [TOOL: name, param: value, ...] lines from the response.
    Returns (tool_calls, clean_text) where tool_calls is a list of {"name": str, "arguments": dict}
    and clean_text is the response with tool call lines removed.
    """
    tool_calls: list[dict[str, Any]] = []
    lines_to_remove: list[tuple[int, int]] = []

    # Match [TOOL: tool_name, key: value, ...] — value can be quoted or unquoted
    tool_pattern = re.compile(
        r'\[TOOL:\s*(\w+)'
        r'((?:\s*,\s*\w+\s*:\s*(?:'
        r'"(?:[^"\\]|\\.)*"'  # double-quoted value
        r"|'(?:[^'\\]|\\.)*'"  # single-quoted value
        r'|[^,\]]+'  # unquoted value
        r'))*)\s*\]',
        re.IGNORECASE
    )

    for m in tool_pattern.finditer(response_text):
        name = m.group(1).strip()
        params_str = m.group(2).strip()
        args: dict[str, Any] = {}

        if params_str:
            # Parse key: value pairs
            param_pattern = re.compile(
                r'(\w+)\s*:\s*('
                r'"(?:[^"\\]|\\.)*"'
                r"|'(?:[^'\\]|\\.)*'"
                r'|[^,\]]+'
                r')'
            )
            for param_m in param_pattern.finditer(params_str):
                key = param_m.group(1).strip()
                val = param_m.group(2).strip()
                # Strip quotes
                if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                    val = val[1:-1]
                args[key] = val

        if name:
            tool_calls.append({"name": name, "arguments": args})
            lines_to_remove.append((m.start(), m.end()))

    # Remove tool call lines from text (reverse order to preserve indices)
    clean_text = response_text
    for start, end in reversed(lines_to_remove):
        clean_text = clean_text[:start] + clean_text[end:]

    return tool_calls, clean_text.strip()


def has_tool_calls(text: str) -> bool:
    """Quick check whether the text contains a ```json action block."""
    return "```json" in text


def format_tool_result_message(tool_call_id: str, content: str) -> str:
    """Format a tool result as natural language for the backend."""
    return f"[TOOL RESULT] Tool '{tool_call_id}' returned: {content}\n\nBased on this result, please continue answering the user's question."


def merge_system_with_tools(system: Any, tool_instructions: str) -> str:
    """Merge an existing system prompt fragment with tool instructions."""
    parts: list[str] = []
    if system:
        s = str(system).strip()
        if s:
            parts.append(s)
    if tool_instructions:
        parts.append(tool_instructions)
    return "\n\n".join(parts)


# ==================== Original completion helpers (unchanged) ====================


def completion_chunk(model: str, delta: dict[str, Any], finish_reason: str | None = None, completion_id: str = "", created: int | None = None) -> dict[str, Any]:
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def completion_response(
    model: str,
    content: str,
    created: int | None = None,
    messages: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    prompt_tokens = count_message_tokens(messages, model) if messages else 0
    completion_tokens = count_text_tokens(content, model) if messages else 0
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_text_chat_completion(backend, messages: list[dict[str, Any]], model: str) -> Iterator[dict[str, Any]]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    request = ConversationRequest(model=model, messages=messages)
    for delta_text in stream_text_deltas(backend, request):
        if not sent_role:
            sent_role = True
            yield completion_chunk(model, {"role": "assistant", "content": delta_text}, None, completion_id, created)
        else:
            yield completion_chunk(model, {"content": delta_text}, None, completion_id, created)
    if not sent_role:
        yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


def collect_chat_content(chunks: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        content = str(delta.get("content") or "")
        if content:
            parts.append(content)
    return "".join(parts)


def chat_messages_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        return [message for message in messages if isinstance(message, dict)]
    prompt = str(body.get("prompt") or "").strip()
    if prompt:
        return [{"role": "user", "content": prompt}]
    raise HTTPException(status_code=400, detail={"error": "messages or prompt is required"})


def chat_image_args(body: dict[str, Any]) -> tuple[str, str, int, list[tuple[bytes, str, str]]]:
    model = str(body.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    prompt = extract_chat_prompt(body)
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    images = [
        (data, f"image_{idx}.png", mime)
        for idx, (data, mime) in enumerate(extract_chat_image(body), start=1)
    ]
    return model, prompt, parse_image_count(body.get("n")), images


def text_chat_parts(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = normalize_messages(chat_messages_from_body(body))
    return model, messages


def image_result_content(result: dict[str, Any]) -> str:
    data = result.get("data")
    if isinstance(data, list) and data:
        return build_chat_image_markdown_content(result)
    return str(result.get("message") or "Image generation completed.")


def image_chat_response(body: dict[str, Any]) -> dict[str, Any]:
    model, prompt, n, images = chat_image_args(body)
    result = collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        response_format="b64_json",
        images=encode_images(images) or None,
    )))
    return completion_response(model, image_result_content(result), int(result.get("created") or 0) or None)


def image_chat_events(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    model, prompt, n, images = chat_image_args(body)
    image_outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        response_format="b64_json",
        images=encode_images(images) or None,
    ))
    yield from stream_image_chat_completion(image_outputs, model)


def stream_image_chat_completion(image_outputs: Iterable[ImageOutput], model: str) -> Iterator[dict[str, Any]]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    sent_text = ""
    for output in image_outputs:
        content = ""
        if output.kind == "progress":
            content = output.text
            sent_text += content
        elif output.kind == "result":
            content = build_chat_image_markdown_content({"data": output.data})
        elif output.kind == "message":
            content = output.text[len(sent_text):] if output.text.startswith(sent_text) else output.text
        if not content:
            continue
        if not sent_role:
            sent_role = True
            yield completion_chunk(model, {"role": "assistant", "content": content}, None, completion_id, created)
        else:
            yield completion_chunk(model, {"content": content}, None, completion_id, created)
    if not sent_role:
        yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


# ==================== Tool-aware wrappers ====================


def _extract_tool_result_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool_result messages and tool_calls to plain text for the backend.
    Messages with role='tool' get converted to user messages.
    Messages with tool_calls (assistant) get flattened to text.
    """
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")

        # OpenAI tool result message: role="tool", tool_call_id=..., content=...
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            text = format_tool_result_message(tool_call_id, str(content))
            result.append({"role": "user", "content": text})
            continue

        if role == "assistant" and msg.get("tool_calls") is not None:
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, (list, tuple)):
                tool_calls = [tool_calls]
            content = msg.get("content", "") or ""
            for tc in tool_calls:
                tc_lines = []
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        name = fn.get("name", "")
                        args = fn.get("arguments", "{}")
                        tc_lines.append(f"[TOOL: {name}, arguments: {args}]")
                tc_text = "\n".join(tc_lines)
                flat_content = (content + "\n" + tc_text).strip() if content else tc_text
                if flat_content:
                    result.append({"role": "assistant", "content": flat_content})
            continue

        result.append(msg)

    return result


def _stream_chat_with_tool_support(
    backend,
    messages: list[dict[str, Any]],
    model: str,
    tool_calls_list: list[dict[str, Any]],
    finish_reason_out: list[str],
) -> Iterator[dict[str, Any]]:
    """Stream chat completion, emitting tool_calls in OpenAI format when detected.
    Accumulated in tool_calls_list and finish_reason_out (as mutable containers).
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # We'll accumulate full text to detect tool calls at the end,
    # but also stream deltas for non-tool-call content.
    full_text = ""
    request = ConversationRequest(model=model, messages=messages)

    for delta_text in stream_text_deltas(backend, request):
        full_text += delta_text
        yield completion_chunk(model, {"role": "assistant", "content": delta_text}, None, completion_id, created)
        created = int(time.time())  # refresh for subsequent chunks

    # Parse tool calls from full text
    parsed_calls, clean_text = parse_tool_calls(full_text)

    if parsed_calls:
        finish_reason_out.append("tool_calls")
        # Build OpenAI-format tool_calls
        tc_list = []
        for i, tc in enumerate(parsed_calls):
            tc_id = f"call_{uuid.uuid4().hex[:24]}"
            tool_calls_list.append(tc)
            tc_list.append({
                "index": i,
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            })
        # Emit a final chunk with tool_calls
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": tc_list}, "finish_reason": "tool_calls"}],
        }
    else:
        finish_reason_out.append("stop")
        yield completion_chunk(model, {}, "stop", completion_id)


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    # Check for image chat first (no tool support for images yet)
    if body.get("stream"):
        if is_image_chat_request(body):
            return image_chat_events(body)
    else:
        if is_image_chat_request(body):
            return image_chat_response(body)

    model = str(body.get("model") or "auto").strip() or "auto"
    raw_messages = chat_messages_from_body(body)

    # Extract tool-related fields from the request
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    # Check if any messages are tool results (role="tool") and convert them
    messages = _extract_tool_result_messages(raw_messages)

    # Build tool instructions and inject into system prompt
    tool_instructions = build_tool_instructions(tools, tool_choice)
    if tool_instructions:
        # Prepend tool instructions as a system message
        messages = [{"role": "system", "content": tool_instructions}] + messages

    if body.get("stream"):
        # Streaming mode — collect tool calls and emit them
        tool_calls_accum: list[dict[str, Any]] = []
        finish_reason_accum: list[str] = []
        return _stream_chat_with_tool_support(
            text_backend(), messages, model, tool_calls_accum, finish_reason_accum
        )

    # Non-streaming mode
    request = ConversationRequest(model=model, messages=messages)
    full_text = collect_text(text_backend(), request)

    # Parse tool calls
    parsed_calls, clean_text = parse_tool_calls(full_text)

    if parsed_calls:
        # Build OpenAI-format tool_calls for the response
        tc_list = []
        for tc in parsed_calls:
            tc_list.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            })
        return completion_response(
            model,
            clean_text,
            messages=messages,
            tool_calls=tc_list,
            finish_reason="tool_calls",
        )

    return completion_response(model, full_text, messages=messages)
