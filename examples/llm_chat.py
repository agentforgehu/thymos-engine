"""Chat with a real LLM while Thymos keeps the long-term memory.

Zero third-party dependencies.  Two providers are supported:

  * DeepSeek  — OpenAI-compatible ``/chat/completions``
  * Ollama    — native ``/api/chat``

The important boundary is still three calls:

    user text -> AgentMemory.context() -> LLM -> AgentMemory.observe()

Quick start
-----------

DeepSeek (PowerShell)::

    $env:DEEPSEEK_API_KEY = "sk-..."
    python examples/llm_chat.py --provider deepseek

Ollama::

    ollama serve
    ollama pull qwen2.5:7b
    python examples/llm_chat.py --provider ollama --model qwen2.5:7b

Non-interactive demo (recommended first)::

    python examples/llm_chat.py --demo --provider ollama

The state file is ``.thymos/llm_chat.json``.  Use ``--fresh`` to start over,
or ``--state`` to keep one file per user/persona.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from thymos import AgentMemory, Profile, ThymosConfig
from thymos.types import MemoryItem

DEFAULT_STATE = ".thymos/llm_chat.json"
DEMO_STATE = ".thymos/llm_chat_demo.json"

SYSTEM_PROMPT = """你是一个有长期记忆的对话助手。

下面会给你「相关记忆」和「当前模拟状态」。请自然地使用它们：
- 只使用确实出现的记忆，不要补写、不要编造细节；
- 不要逐条复述记忆，像正常人一样接话；
- 如果记忆和用户当前说法冲突，以用户当前说法为准，并简短确认；
- 这些是模拟状态，不要声称自己有真实的身体、激素或医学感受；
- 不知道就说不知道，不要用记忆之外的信息假装记得。"""


class LLMError(RuntimeError):
    """A provider/network error that the CLI can explain to the user."""


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _normalize_base(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    if not value:
        raise LLMError("base URL 不能为空")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


def _endpoint(base: str, path: str) -> str:
    base = _normalize_base(base)
    if base.endswith(path):
        return base
    return base + path


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "thymos-llm-chat/0.1",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise LLMError(f"HTTP {exc.code} from {url}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"无法连接 {url}：{exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMError(f"请求超时（{timeout:g}s）：{url}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"服务返回的不是 JSON（{url}）：{raw[:300]}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"服务返回了意外的 JSON 类型：{type(data).__name__}")
    return data


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class DeepSeekClient:
    name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self.api_key = api_key
        self.base_url = _normalize_base(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = _endpoint(self.base_url, "/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        data = _post_json(
            url,
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if data.get("error"):
            raise LLMError(f"DeepSeek 返回错误：{data['error']}")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"无法解析 DeepSeek 响应：{str(data)[:500]}") from exc
        content = (content or "").strip()
        if not content:
            raise LLMError("DeepSeek 返回了空内容")
        return content


class OllamaClient:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self.base_url = _normalize_base(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = _endpoint(self.base_url, "/api/chat")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        data = _post_json(url, payload, timeout=self.timeout)
        if data.get("error"):
            raise LLMError(f"Ollama 返回错误：{data['error']}")
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"无法解析 Ollama 响应：{str(data)[:500]}") from exc
        content = (content or "").strip()
        if not content:
            raise LLMError("Ollama 返回了空内容")
        return content


def build_client(args: argparse.Namespace) -> DeepSeekClient | OllamaClient:
    provider = args.provider
    if provider == "auto":
        provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "ollama"

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise LLMError(
                "未设置 DEEPSEEK_API_KEY。\n"
                "PowerShell: $env:DEEPSEEK_API_KEY = 'sk-...'\n"
                "或改用: python examples/llm_chat.py --provider ollama"
            )
        base_url = args.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        return DeepSeekClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

    base_url = args.base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = args.model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    return OllamaClient(
        base_url=base_url,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


# --------------------------------------------------------------------------- #
# Conversation loop
# --------------------------------------------------------------------------- #
def _trim_history(history: list[dict[str, str]], max_messages: int) -> None:
    """Keep a short rolling window; long-term continuity belongs to Thymos."""
    if max_messages <= 0:
        history.clear()
        return
    while len(history) > max_messages:
        history.pop(0)
    if history and history[0]["role"] == "assistant":
        history.pop(0)


def run_turn(
    memory: AgentMemory,
    client: DeepSeekClient | OllamaClient,
    history: list[dict[str, str]],
    user_text: str,
    args: argparse.Namespace,
    *,
    remember: bool = True,
    show_context: bool = False,
) -> tuple[str, MemoryItem | None]:
    context = memory.context(
        user_text,
        top_k=args.top_k,
        min_score=args.min_score,
        # Surfacing a memory in the prompt is not the same as the user
        # rehearsing it.  Default to read-only; opt in with --review.
        review=args.review,
    )
    if show_context or args.show_context:
        print("\n--- 记忆与状态（LLM 实际看到的部分）---")
        print(context)
        print("--- end ---")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": args.system + "\n\n" + context}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    reply = client.chat(messages)

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    _trim_history(history, args.history)

    item = memory.observe(user_text) if remember else None
    return reply, item


def _print_memory(item: MemoryItem | None) -> None:
    if item is None:
        print("  [memory] 跳过：没有达到重要性阈值，或这是问句")
        return
    print(
        f"  [memory] 写入：S={item.stability:.2f}d  "
        f"salience={item.salience:.2f}  {item.content[:48]}"
    )


def _print_stats(memory: AgentMemory) -> None:
    stats = memory.mind.stats()
    print(
        f"  [state] 记忆 {stats.get('total', len(memory.mind.store))} 条，"
        f"平均可提取度 {stats.get('avg_retrievability', 0):.2f}，"
        f"心境 {memory.mind.affect.state.mood_valence:+.2f}"
    )


def run_demo(memory: AgentMemory, client: DeepSeekClient | OllamaClient, args: argparse.Namespace) -> None:
    print("=" * 72)
    print("DAY 1 · 先让用户说三件事")
    print("=" * 72)
    history: list[dict[str, str]] = []
    facts = [
        "我不吃虾，过敏",
        "我喜欢冰美式，不加糖",
        "我周四面试，有点紧张",
    ]
    for text in facts:
        reply, item = run_turn(memory, client, history, text, args, remember=True)
        print(f"\n你：{text}")
        print(f"助手：{reply}")
        _print_memory(item)
    _print_stats(memory)

    print("\n" + "=" * 72)
    print("30 DAYS LATER · 清空对话历史，只让 Thymos 提供记忆")
    print("=" * 72)
    memory.mind.tick(30 * 86400)
    memory.end_session(7.5)
    history.clear()
    _print_stats(memory)

    questions = [
        "我不吃什么？",
        "我面试是哪天？",
        "我喜欢喝什么？",
    ]
    for question in questions:
        reply, item = run_turn(
            memory,
            client,
            history,
            question,
            args,
            remember=False,
            show_context=True,
        )
        print(f"\n你：{question}")
        print(f"助手：{reply}")
        _print_memory(item)

    memory.save()
    print("\n" + "=" * 72)
    print("状态已保存：", Path(args.state).resolve())
    print("提示：把 --provider 换成 deepseek/ollama，就是同一个 Bot 的真实 LLM 版本。")
    print("=" * 72)


def run_interactive(
    memory: AgentMemory, client: DeepSeekClient | OllamaClient, args: argparse.Namespace
) -> None:
    print("=" * 72)
    print(f"Thymos × {client.name}  |  model={client.model}")
    print(f"状态文件：{Path(args.state).resolve()}")
    print("输入 exit / quit / 退出 结束；每次退出会做 7.5 小时睡眠巩固并保存。")
    print("=" * 72)

    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q", "退出", "结束"}:
            break
        try:
            reply, item = run_turn(memory, client, history, user_text, args)
        except LLMError as exc:
            print(f"\n[LLM 错误] {exc}")
            continue
        print(f"助手：{reply}")
        _print_memory(item)

    result = memory.end_session(7.5)
    print(
        f"\n[session] 已巩固 {result.get('replayed', 0)} 条记忆，"
        f"保存到 {Path(args.state).resolve()}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 LLM（DeepSeek / Ollama）驱动 Thymos 记忆。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "deepseek", "ollama"],
        default=os.environ.get("THYMOS_LLM_PROVIDER", "auto"),
        help="auto: 有 DEEPSEEK_API_KEY 就用 DeepSeek，否则用 Ollama",
    )
    parser.add_argument("--model", default=None, help="模型名；默认 deepseek-chat / qwen2.5:7b")
    parser.add_argument("--base-url", default=None, help="覆盖 API 地址")
    parser.add_argument("--state", default=DEFAULT_STATE, help="每个用户/人格一个状态文件")
    parser.add_argument("--fresh", action="store_true", help="启动前删除状态文件")
    parser.add_argument("--demo", action="store_true", help="跑脚本化的 30 天演示，而不是交互")
    parser.add_argument("--system", default=SYSTEM_PROMPT, help="系统提示词")
    parser.add_argument("--user-name", default="朋友", help="Profile 名称")
    parser.add_argument("--location", default="上海", help="Profile 地点（离线模式下只做上下文）")
    parser.add_argument("--top-k", type=int, default=4, help="注入几条记忆")
    parser.add_argument("--min-score", type=float, default=0.30, help="低于此分数的记忆不注入")
    parser.add_argument(
        "--review",
        action="store_true",
        help="把注入的 Top-k 视为一次复习（会提高稳定度；默认关闭）",
    )
    parser.add_argument("--history", type=int, default=8, help="保留多少条短期对话消息（0=不保留）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--show-context", action="store_true", help="每轮打印 LLM 看到的记忆块")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo and args.state == DEFAULT_STATE:
        args.state = DEMO_STATE
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh or (args.demo and args.state == DEMO_STATE):
        state_path.unlink(missing_ok=True)

    try:
        client = build_client(args)
    except LLMError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    memory = AgentMemory(
        str(state_path),
        profile=Profile(name=args.user_name, location_name=args.location),
        config=ThymosConfig(offline=True, seed=7),
    )

    print(f"Provider: {client.name}  |  model: {client.model}  |  base: {client.base_url}")
    try:
        if args.demo:
            run_demo(memory, client, args)
        else:
            run_interactive(memory, client, args)
    except LLMError as exc:
        print(f"\n[LLM 错误] {exc}", file=sys.stderr)
        if client.name == "ollama":
            print("提示：确认 `ollama serve` 已启动，且模型已 `ollama pull`。", file=sys.stderr)
        memory.save()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
