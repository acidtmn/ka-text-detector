import asyncio
import hmac
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from .registry import list_models

pool = None
running = 0
extracting = 0
warm = False
token = os.environ.get("DETECTOR_SERVICE_TOKEN", "")


def start_pool():
    return ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    )


def stop_pool():
    global pool
    if pool:
        for process in list((getattr(pool, "_processes", {}) or {}).values()):
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        pool = None


@asynccontextmanager
async def lifespan(app):
    global pool, warm
    if len(token) < 32:
        raise RuntimeError("DETECTOR_SERVICE_TOKEN must be provisioned locally")
    pool = start_pool()
    try:
        await asyncio.to_thread(list_models)
        from .engine import warmup

        await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(pool, warmup, "upstream-v2"), 90
        )
        warm = True
    except Exception:
        warm = False
    yield
    stop_pool()


app = FastAPI(
    title="KA Text Detector / local research",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.url.path != "/health" and not hmac.compare_digest(
        request.headers.get("authorization", ""), f"Bearer {token}"
    ):
        return JSONResponse({"error": "unauthorized"}, 401)
    if request.method == "POST":
        size = request.headers.get("content-length", "")
        if not size.isdigit() or int(size) > 6 * 1024 * 1024:
            return JSONResponse({"error": "request_size_limit"}, 413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    return JSONResponse({"error": "invalid_input"}, 422)


@app.get("/health")
def health():
    return {"status": "ok", "warm": warm}


@app.get("/v1/status")
def status():
    from .engine import DISCLAIMER

    current = None
    try:
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    except OSError:
        pass
    return {
        "ready": warm,
        "models": list_models(),
        "running": running,
        "maxParallel": 2,
        "memoryMb": round(current / 1024**2, 1) if current else None,
        "offline": True,
        "disclaimer": DISCLAIMER,
    }


class AnalyzeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=100, max_length=60000)
    models: list[str] = Field(min_length=1, max_length=2)
    maxParallel: int = Field(default=1, ge=1, le=2)
    timeoutSeconds: int = Field(default=60, ge=5, le=90)


@app.post("/v1/analyze")
async def analyze(request: Request, body: AnalyzeInput):
    global running, pool, warm
    if len(set(body.models)) != len(body.models):
        return JSONResponse({"error": "duplicate_model"}, 422)
    available = {m["id"] for m in list_models() if m["ready"]}
    if any(m not in available for m in body.models):
        return JSONResponse({"error": "model_unavailable"}, 409)
    if running >= body.maxParallel:
        return JSONResponse({"error": "service_busy"}, 429)
    running += 1
    try:
        from .engine import analyze as run_analysis

        if pool is None:
            pool = start_pool()
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                pool, run_analysis, body.text, body.models
            ),
            body.timeoutSeconds,
        )
        warm = True
        return {"version": 1, "data": result}
    except asyncio.TimeoutError:
        stop_pool()
        warm = False
        return JSONResponse({"error": "analysis_timeout"}, 504)
    except ValueError:
        return JSONResponse({"error": "analysis_not_applicable"}, 422)
    except Exception:
        stop_pool()
        warm = False
        return JSONResponse({"error": "analysis_failed"}, 503)
    finally:
        running -= 1


@app.post("/v1/extract")
async def extract(file: UploadFile = File(...)):
    global extracting
    if extracting >= 2:
        await file.close()
        return JSONResponse({"error": "service_busy"}, 429)
    extracting += 1
    try:
        return await extract_document(file)
    finally:
        extracting -= 1


async def extract_document(file):
    extension = Path(file.filename or "").suffix.lower().lstrip(".")
    if extension not in ["txt", "docx", "pdf"]:
        return JSONResponse({"error": "unsupported_format"}, 415)
    data = await file.read(5 * 1024 * 1024 + 1)
    await file.close()
    if len(data) > 5 * 1024 * 1024:
        return JSONResponse({"error": "file_too_large"}, 413)
    with tempfile.TemporaryDirectory(prefix="ka-extract-") as directory:
        path = Path(directory) / f"document.{extension}"
        path.write_bytes(data)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "ka_detector.documents",
                str(path),
                extension,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=12)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return JSONResponse({"error": "document_timeout"}, 422)
            result = json.loads(output)
            if process.returncode != 0 or "error" in result:
                return JSONResponse(
                    {"error": result.get("error", "invalid_document")}, 422
                )
            return {"version": 1, "data": result}
        except Exception:
            return JSONResponse({"error": "invalid_document"}, 422)
