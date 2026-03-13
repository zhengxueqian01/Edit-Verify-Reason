import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError


def load_image_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def extract_braced(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\{([^{}]+)\}", text or "")]


def normalize(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b\d{4}\b", text or "")


def is_match(model_output: str, expected_answer: str) -> bool:
    if not expected_answer:
        return False
    expected = expected_answer.strip()
    output = (model_output or "").strip()
    expected_years = extract_years(expected)
    if expected_years:
        output_years = extract_years(output)
        if not output_years:
            return False
        return output_years[0] == expected_years[0]
    if expected and expected in output:
        return True
    expected_parts = extract_braced(expected)
    if expected_parts:
        return any(part in output for part in expected_parts)
    return False


def is_max_question(question: str) -> bool:
    return "overall maximum" in (question or "").lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch test task1-mix-area add-change samples."
    )
    parser.add_argument("--limit", type=int, default=0, help="Max samples to test; 0 for all.")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--base-url", default="https://aihubmix.com/v1")
    parser.add_argument("--data-dir", default="task1-mix-area/add-change")
    parser.add_argument("--out", default="add_change_results.txt")
    parser.add_argument("--image-suffix", default=".png")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds per request.")
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries on 429.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("AIHUBMIX_API_KEY_ZZT", "")
    if not api_key:
        raise SystemExit("AIHUBMIX_API_KEY_ZZT is required.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    tested = 0
    matched = 0
    max_tested = 0
    max_matched = 0

    out_path = Path(args.out)
    if not out_path.is_absolute():
        if out_path.parts and out_path.parts[0] == "evaluation":
            out_path = repo_root / out_path
        else:
            out_path = eval_dir / out_path

    out_file = out_path.open("w", encoding="utf-8")
    try:
        first_io_written = False
        for sub in sorted(data_dir.iterdir()):
            if not sub.is_dir() or not sub.name.isdigit():
                continue
            sample_path = sub / f"{sub.name}.json"
            if not sample_path.exists():
                continue
            with sample_path.open("r", encoding="utf-8") as f:
                sample = json.load(f)

            qa_list = sample.get("QA") or []
            if not qa_list:
                continue

            image_path = sub / f"{sub.name}{args.image_suffix}"
            if not image_path.exists():
                msg = f"[WARN] image missing: {image_path}"
                print(msg)
                out_file.write(msg + "\n")
                continue

            image_b64 = load_image_base64(image_path)
            image_url = f"data:image/{image_path.suffix.lstrip('.').lower()};base64,{image_b64}"

            for qa in qa_list:
                question = qa.get("question")
                if not question:
                    continue
                expected = qa.get("answer", "")
                context_parts = []
                if sample.get("operation_target"):
                    context_parts.append(
                        "Operation target: "
                        + json.dumps(sample.get("operation_target"), ensure_ascii=False)
                    )
                if sample.get("data_change"):
                    context_parts.append(
                        "Data change: "
                        + json.dumps(sample.get("data_change"), ensure_ascii=False)
                    )
                prompt = question
                if context_parts:
                    prompt = question + "\n\n" + "\n".join(context_parts)
                prompt += (
                    "\n\nFirst provide the year answer only (e.g., 2019). "
                    "Then provide a 1-sentence rationale."
                )

                attempt = 0
                while True:
                    try:
                        response = client.chat.completions.create(
                            model=args.model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": image_url}},
                                    ],
                                }
                            ],
                        )
                        break
                    except RateLimitError:
                        attempt += 1
                        if attempt > args.max_retries:
                            raise
                        backoff = min(60.0, (2 ** (attempt - 1)))
                        print(f"[WARN] 429 rate limit, retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                time.sleep(args.sleep)

                output = response.choices[0].message.content or ""
                if not first_io_written:
                    out_file.write("FIRST_CASE_INPUT:\n")
                    out_file.write(prompt + "\n")
                    out_file.write(f"FIRST_CASE_IMAGE_PATH: {image_path}\n")
                    out_file.write("FIRST_CASE_OUTPUT:\n")
                    out_file.write(output + "\n")
                    out_file.write("=" * 40 + "\n")
                    first_io_written = True
                ok = is_match(output, expected)
                if ok:
                    matched += 1
                if is_max_question(question):
                    max_tested += 1
                    if ok:
                        max_matched += 1

                line1 = f"[{sample.get('id')}] Q: {question}"
                line2 = f"Expected: {expected}"
                line3 = f"Answer: {output}"
                line4 = f"Match: {'yes' if ok else 'no'}"
                sep = "-" * 40
                print(line1)
                print(line2)
                print(line3)
                print(line4)
                print(sep)
                out_file.write(line1 + "\n")
                out_file.write(line2 + "\n")
                out_file.write(line3 + "\n")
                out_file.write(line4 + "\n")
                out_file.write(sep + "\n")

                tested += 1
                if args.limit and tested >= args.limit:
                    summary = f"Matched {matched}/{tested}"
                    max_line = f"Max: {max_matched}/{max_tested}"
                    print(summary)
                    print(max_line)
                    out_file.write(summary + "\n")
                    out_file.write(max_line + "\n")
                    return

        if tested:
            accuracy = matched / tested * 100
            summary = f"Matched {matched}/{tested}"
            acc_line = f"Accuracy: {accuracy:.2f}%"
            max_line = f"Max: {max_matched}/{max_tested}"
            print(summary)
            print(acc_line)
            print(max_line)
            out_file.write(summary + "\n")
            out_file.write(acc_line + "\n")
            out_file.write(max_line + "\n")
    finally:
        out_file.close()


if __name__ == "__main__":
    main()
