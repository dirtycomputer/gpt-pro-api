#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_FILE = REPO_ROOT / "request.txt"
RESPONSE_FILE = REPO_ROOT / "response.txt"
HISTORY_DIR = REPO_ROOT / "history"
HISTORY_FILE = HISTORY_DIR / "history.jsonl"

MODEL = os.getenv("MODEL") or "gpt-5-pro-2025-10-06"  # 建议用 Models 页面里的当前可用模型


def load_last_response_id() -> str | None:
    if not HISTORY_FILE.exists():
        return None
    try:
        # 读取最后一行（最近的一次对话）
        *_, last = HISTORY_FILE.read_text(encoding="utf-8").splitlines() or [None]
        if not last:
            return None
        rec = json.loads(last)
        return rec.get("response_id")
    except Exception:
        return None


def append_history(record: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    if not REQUEST_FILE.exists():
        raise SystemExit("request.txt 不存在。请先在仓库根目录创建它并写入要提问的内容。")

    question = REQUEST_FILE.read_text(encoding="utf-8").strip()
    if not question:
        raise SystemExit("request.txt 为空。请写入你的问题后重试。")

    client = OpenAI()  # 使用环境变量 OPENAI_API_KEY 进行鉴权

    previous_id = load_last_response_id()

    kwargs = {
        "model": MODEL,
        "input": question,
    }
    if previous_id:
        # 将本次问题与上一次响应形成“串联”以保留上下文
        kwargs["previous_response_id"] = previous_id

    resp = client.responses.create(**kwargs)

    # 提取纯文本输出（官方 SDK 暴露 output_text）
    answer_text = getattr(resp, "output_text", None)
    if not answer_text:
        # 兼容：如果 SDK 版本无 output_text，尝试从结构化输出中拼接
        try:
            chunks = []
            for item in resp.output:  # type: ignore[attr-defined]
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            chunks.append(c.get("text", ""))
            answer_text = "\n".join(chunks).strip()
        except Exception:
            answer_text = ""  # 保底

    RESPONSE_FILE.write_text(answer_text, encoding="utf-8")

    # 记录历史（request/response/模型/响应ID/用量等）
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "request": question,
        "response": answer_text,
        "response_id": getattr(resp, "id", None),
        "usage": getattr(resp, "usage", None) and resp.usage.__dict__,
    }
    append_history(record)

    print("已写入:", RESPONSE_FILE)
    print("已追加历史:", HISTORY_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())