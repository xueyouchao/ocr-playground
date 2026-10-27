"""Any OpenAI-compatible VLM endpoint: one page image in, markdown out.

Works for a vLLM server (GLM-OCR, Chandra OCR 2, Unlimited-OCR, Nemotron-parse all serve that way)
or a provider's hosted API, as long as it accepts image_url content.

    python scripts/runners/vlm_endpoint.py --id glm-ocr --model glm-ocr \
        --base-url https://.../v1 --api-key "$KEY"
"""
import argparse
import base64
import json
import os
import urllib.error
import urllib.request

from _common import page_paths, write

# Each family names its own instruction; "Text Recognition:" is what GLM-OCR's recipe sends, and a
# generic "convert to markdown" prompt measurably changes what the smaller VLMs emit.
PROMPT = "Convert this page to markdown. Keep tables as HTML tables and preserve reading order."


def ask(base_url: str, model: str, api_key: str, image: bytes, prompt: str, max_tokens: int,
        timeout: int = 900) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    # Modal answers a cold endpoint with a 303 carrying an attempt token. Following it by hand is the
    # cold-start wait: the second request parks until the container is up.
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for _attempt in range(4):
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            if error.code in (303, 307, 308) and error.headers.get("Location"):
                url = error.headers["Location"]
                continue
            raise SystemExit(f"{base_url} said {error.code}: {error.read()[:400]!r}")
    raise SystemExit(f"{base_url} redirected too many times")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="system id from systems.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not args.base_url:
        raise SystemExit("no endpoint: pass --base-url or set VLM_BASE_URL")

    pages = {}
    for page in page_paths():
        for attempt in range(3):
            try:
                pages[page.stem] = ask(args.base_url, args.model, args.api_key, page.read_bytes(),
                                       args.prompt, args.max_tokens)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                # a long generation can lose the socket; the work is per page, so retry just this page
                print(f"{page.stem}: attempt {attempt + 1} lost the connection ({error})", flush=True)
                if attempt == 2:
                    raise SystemExit(f"{page.stem} failed three times")
        print(f"{page.stem}: {len(pages[page.stem])} chars", flush=True)
        write(args.id, pages)  # keep progress: a dropped run leaves whole pages behind it
    write(args.id, pages)


if __name__ == "__main__":
    main()
