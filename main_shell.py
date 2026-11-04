
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, message_chunk_to_message
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from peas_agent_runtime import SessionStore
from peas_agent_skills import load_all_skills
from peas_agent_tools import get_builtin_tools

load_dotenv()

DEFAULT_SESSION_PATH = "sessions/session.jsonl"
StreamKind = Literal["reasoning", "text"]


@tool
def calculator(a: float, b: float, operation: str) -> float:
    """對兩個數字做加減乘除運算。需要計算時請呼叫此工具。"""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b != 0:
            return a / b
        return "Error: Division by zero"
    return "Error: Unsupported operation"


def _reasoning_text_from_block(block: dict[str, Any]) -> str:
    reasoning = block.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    summary = block.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    if isinstance(summary, list):
        parts: list[str] = []
        for item in summary:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "".join(parts)
    return ""


def _text_from_block(block: dict[str, Any]) -> str:
    text = block.get("text")
    return text if isinstance(text, str) else ""


def iter_content_blocks(message: BaseMessage | AIMessageChunk) -> Iterable[dict[str, Any]]:
    blocks = getattr(message, "content_blocks", None)
    if blocks:
        for block in blocks:
            if isinstance(block, dict):
                yield block
        return

    content = message.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block


def iter_chunk_deltas(chunk: AIMessageChunk) -> Iterable[tuple[StreamKind, str]]:
    saw_blocks = False
    for block in iter_content_blocks(chunk):
        saw_blocks = True
        block_type = block.get("type")
        if block_type == "reasoning":
            text = _reasoning_text_from_block(block)
            if text:
                yield "reasoning", text
        elif block_type == "text":
            text = _text_from_block(block)
            if text:
                yield "text", text

    if saw_blocks:
        return

    content = chunk.content
    if isinstance(content, str) and content:
        yield "text", content


def extract_answer_text(message: BaseMessage) -> str:
    if not isinstance(message, AIMessage):
        content = message.content
        return content.strip() if isinstance(content, str) else str(content).strip()

    parts: list[str] = []
    saw_blocks = False
    for block in iter_content_blocks(message):
        saw_blocks = True
        if block.get("type") == "text":
            text = _text_from_block(block)
            if text:
                parts.append(text)

    if saw_blocks:
        return "".join(parts).strip()

    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = _text_from_block(block)
                if text:
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def build_system_prompt(host_context: str | None = None) -> dict:
    soul = "你是法鬥超人，一位擅長引導學生python問題的助教"
    user = (
        "我的名字叫Ray, 是一位學生。"
        "我喜歡你用簡潔扼要的方式引導我解決問題，並且在回答中加入一些幽默感。"
    )
    skills_text = load_all_skills()
    system_content = f"""
    # 角色設定
    {soul}
    # 使用者設定
    {user}
    # 技能卷軸
    {skills_text}
    # Host Environment
    {host_context or ""}
    """
    return {"role": "system", "content": system_content}


class Agent:
    def __init__(self, llm, tools, session_path=None, host_context=None):
        self.llm = llm.bind_tools(tools)
        self.tool_map = {t.name: t for t in tools}
        self.host_context = host_context
        self.session = SessionStore(session_path)
        self.history = self.session.load()

    def _stream_response(
        self,
        messages: list[dict[str, Any]],
        *,
        on_token: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> AIMessage:
        acc: AIMessageChunk | None = None
        for chunk in self.llm.stream(messages):
            acc = chunk if acc is None else acc + chunk
            for kind, delta in iter_chunk_deltas(chunk):
                if not delta:
                    continue
                if kind == "reasoning":
                    if on_reasoning is not None:
                        on_reasoning(delta)
                elif on_token is not None:
                    on_token(delta)
        if acc is None:
            raise RuntimeError("模型串流未回傳任何 chunk")
        return message_chunk_to_message(acc)

    def chat(
        self,
        user_text: str,
        image_path=None,
        on_token=None,
        on_reasoning=None,
        on_stream_reset=None,
    ) -> str:
        # image_path：本教學版忽略；簽名保留以相容呼叫端
        system_prompt = build_system_prompt(self.host_context)
        user_prompt = {"role": "user", "content": user_text}
        self.history.append(user_prompt)

        response = self._stream_response(
            [system_prompt, *self.history],
            on_token=on_token,
            on_reasoning=on_reasoning,
        )

        while response.tool_calls:
            if on_stream_reset is not None:
                on_stream_reset()
            self.history.append(
                {
                    "role": "assistant",
                    "content": extract_answer_text(response),
                    "tool_calls": response.tool_calls,
                }
            )

            for tool_call in response.tool_calls:
                print(tool_call)
                name = tool_call["name"]
                if name not in self.tool_map:
                    content = f"Error: Tool '{name}' not found."
                else:
                    content = str(self.tool_map[name].invoke(tool_call["args"]))
                self.history.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call["id"],
                    }
                )

            response = self._stream_response(
                [system_prompt, *self.history],
                on_token=on_token,
                on_reasoning=on_reasoning,
            )

        answer = extract_answer_text(response)
        self.history.append({"role": "assistant", "content": answer})
        self.session.save(self.history)
        return answer


def create_agent(session_path=None, host_context=None):
    llm = ChatOpenAI(
        api_key=os.environ.get("API_KEY"),
        model=os.environ.get("MODEL_NAME"),
        temperature=os.environ.get("TEMPERATURE"),
        base_url=os.environ.get("BASE_URL"),
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "low", "summary": "auto"},
    )
    return Agent(
        llm,
        [*get_builtin_tools(), calculator],
        session_path or DEFAULT_SESSION_PATH,
        host_context,
    )


def main() -> None:
    agent = create_agent()
    while True:
        question = input("\n你的問題: ").strip()
        if not question:
            print("請輸入一個問題。")
            continue
        if question.lower() == "quit":
            print("Exiting...")
            break
        print("\nAI:", agent.chat(question))


if __name__ == "__main__":
    main()
