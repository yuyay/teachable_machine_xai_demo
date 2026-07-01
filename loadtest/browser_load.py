"""Layer-2 load test: N concurrent headless browser sessions against the app.

Each session: open the app -> upload a Teachable Machine model zip -> capture a
(fake) camera photo -> wait for the prediction + heatmap -> assert no error.
Runs N sessions concurrently and reports X/N passed.

Usage:
    .venv/bin/python loadtest/browser_load.py --n 10 \
        --url https://xai-demo-ba4vygvkya-an.a.run.app
"""
import argparse
import asyncio
import os
import tempfile
import zipfile

from playwright.async_api import async_playwright, Browser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "https://xai-demo-ba4vygvkya-an.a.run.app"


def _write_fake_y4m(path: str, width: int = 224, height: int = 224, frames: int = 10) -> None:
    """Write a minimal mid-gray I420 .y4m for Chromium's fake video capture."""
    y = bytes([128]) * (width * height)
    u = bytes([128]) * ((width // 2) * (height // 2))
    v = bytes([128]) * ((width // 2) * (height // 2))
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 C420\n".encode())
        for _ in range(frames):
            f.write(b"FRAME\n")
            f.write(y + u + v)


def _build_model_zip(path: str) -> None:
    """Build a Teachable-Machine-style zip from the repo's temp_model.h5."""
    h5 = os.path.join(REPO_ROOT, "temp_model.h5")
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(h5, "keras_model.h5")
        zf.writestr("labels.txt", "0 ClassA\n1 ClassB\n")


async def _run_session(browser: Browser, url: str, zip_path: str, timeout_ms: int, idx: int) -> dict[str, object]:
    """Run a single browser session: upload model, take photo, check result.

    Args:
        browser: Playwright Browser instance (shared across sessions).
        url: Target app URL.
        zip_path: Path to the pre-built model zip file.
        timeout_ms: Per-step timeout in milliseconds.
        idx: Session index (for logging).

    Returns:
        Dict with keys ``idx``, ``ok`` (bool), ``error`` (str or None).
    """
    res = {"idx": idx, "ok": False, "error": None}
    ctx = None
    try:
        ctx = await browser.new_context()
        try:
            await ctx.grant_permissions(["camera"], origin=url)
        except Exception:  # noqa: BLE001 - fake-ui already auto-grants; ignore
            pass
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # (a) upload the model zip into the (hidden) file input
        await page.locator('input[type="file"]').first.set_input_files(
            zip_path, timeout=timeout_ms
        )
        await page.get_by_text("モデルが正常に読み込まれました").wait_for(timeout=timeout_ms)
        # (b) capture a photo — Streamlit camera_input button label verified against live DOM
        await page.get_by_role("button", name="Take Photo").click(timeout=timeout_ms)
        # (c) wait for prediction result + heatmap
        await page.get_by_text("予測クラス").wait_for(timeout=timeout_ms)
        body = await page.inner_text("body")
        if "AxiosError" in body or "エラーが発生" in body:
            raise RuntimeError("error text present on page")
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - report per-session failure
        res["error"] = repr(exc)
    finally:
        if ctx is not None:
            await ctx.close()
    return res


async def _main_async(n: int, url: str, timeout_ms: int, headed: bool) -> int:
    """Run N concurrent browser sessions and print X/N passed.

    Args:
        n: Number of concurrent sessions.
        url: Target app URL.
        timeout_ms: Per-step timeout in milliseconds.
        headed: If True, launch visible browser windows (debug mode).

    Returns:
        0 if all sessions passed, 1 otherwise.
    """
    with tempfile.TemporaryDirectory(prefix="loadtest_") as tmpdir:
        # One shared fake camera + model zip for all sessions: the .y4m is read
        # once at browser launch, and concurrent read-only set_input_files on the
        # zip is safe across contexts.
        y4m = os.path.join(tmpdir, "fake_camera.y4m")
        zip_path = os.path.join(tmpdir, "model.zip")
        _write_fake_y4m(y4m)
        _build_model_zip(zip_path)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=not headed,
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    f"--use-file-for-fake-video-capture={y4m}",
                ],
            )
            results = await asyncio.gather(
                *[_run_session(browser, url, zip_path, timeout_ms, i) for i in range(n)]
            )
            await browser.close()

    passed = sum(1 for r in results if r["ok"])
    print(f"\n=== {passed}/{n} sessions passed (url={url}) ===")
    for r in results:
        if not r["ok"]:
            print(f"  session {r['idx']} FAILED: {r['error']}")
    return 0 if passed == n else 1


def main() -> int:
    """Entry point: parse args and run the async load test."""
    parser = argparse.ArgumentParser(description="Concurrent browser load test")
    parser.add_argument("--n", type=int, default=10, help="concurrent sessions")
    parser.add_argument("--url", default=DEFAULT_URL, help="target app URL")
    parser.add_argument("--timeout", type=int, default=120, help="per-step timeout (s)")
    parser.add_argument("--headed", action="store_true", help="show browsers (debug)")
    args = parser.parse_args()
    return asyncio.run(_main_async(args.n, args.url, args.timeout * 1000, args.headed))


if __name__ == "__main__":
    raise SystemExit(main())
