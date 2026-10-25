"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --model gemini-3.5-flash --model gemini-3.5-flash-lite \
        --question "LoRAとは？" --question "RAGとは？"

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class Measurement:
    model: str
    question: str
    answer: str
    elapsed_seconds: float
    prompt_tokens: int
    output_tokens: int


def measure(question: str, model: str = MODEL) -> Measurement:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed_seconds = time.perf_counter() - started_at

    usage = response.usage_metadata
    return Measurement(
        model=model,
        question=question,
        answer=response.text,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
    )


def ask(question: str) -> str:
    result = measure(question)
    print(f"[応答時間] {result.elapsed_seconds:.3f}秒")
    print(f"[入力トークン数] {result.prompt_tokens}")
    print(f"[出力トークン数] {result.output_tokens}")

    return result.answer


def compare(models: list[str], questions: list[str]) -> None:
    """複数のモデルと質問を同じ条件で実行し、測定値を比較する。"""
    print("モデル | 質問 | 応答時間(秒) | 入力トークン | 出力トークン")
    print("--- | --- | ---: | ---: | ---:")
    for model in models:
        for question in questions:
            result = measure(question, model)
            print(
                f"{result.model} | {result.question} | "
                f"{result.elapsed_seconds:.3f} | {result.prompt_tokens} | {result.output_tokens}"
            )
            print(f"\n[{result.model} / {result.question}]\n{result.answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="比較するモデル名（複数指定可）。例: --model gemini-3.5-flash",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="比較する質問（複数指定可）。指定時は位置引数より優先する",
    )
    args = parser.parse_args()

    try:
        if args.models or args.questions:
            models = args.models or [MODEL]
            questions = args.questions or [args.question]
            compare(models, questions)
        else:
            print(f"[model] {MODEL}")
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
