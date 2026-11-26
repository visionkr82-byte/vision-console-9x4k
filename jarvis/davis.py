#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자비스 — 윈도우 PC용 음성 비서 (v4 · 가입 없이)

"자비스"라고 부르면 "네, 대표님" 하고 대답한 뒤,
질문을 받아 클로드 코드(Claude Code CLI)로 답을 만들어 음성으로 읽어 줍니다.

부르는 말은 두 가지 방법으로 알아듣습니다.
  1) 오프라인 음성인식(Vosk 한국어 소형 모델) — 빠른 1차 확인
  2) 받아쓰기 모델(Whisper) — 시끄러운 곳에서도 놓치지 않는 2차 확인
둘 중 하나만 알아들어도 대답합니다. (설정: wake_engine)

실행 방법
    python  davis.py --window      작은 상태 창을 띄우고 실행 (기본)
    pythonw davis.py --silent      창 없이 실행 (자동 시작용)
    python  davis.py --console     로그를 화면에 보면서 실행 (문제 확인용)
    python  davis.py --setup       설정 파일 만들기 + 한국어 음성모델 내려받기
    python  davis.py --check       설치 상태 자세히 점검
    python  davis.py --test-wake   부르는 말 인식 테스트 (20초)
    python  davis.py --test-voice  스피커 소리 테스트
    python  davis.py --list-mics   마이크(입력 장치) 목록 보기
    python  davis.py --version     버전 보기

v4 에서 달라진 점 (요약)
    - 프로그램이 조용히 죽지 않습니다: logs\\YYYY-MM-DD.log 에 모두 남기고,
      상태 창에 마지막 인식·오류를 한국어로 보여 줍니다.
    - 시작할 때 자가진단(음성모델 / 마이크 / 클로드 코드)을 하고 결과를 알려 줍니다.
    - 시끄러운 곳에서 Vosk 가 '자비스'를 '서비스'로 잘못 듣던 문제를
      Whisper 2차 확인으로 보완했습니다.
    - 마이크를 직접 고를 수 있고(mic_device), 열 수 없으면 쓸 수 있는 장치를 한글로 보여 줍니다.
    - 말이 끝나면 0.8초 만에 질문을 끊고, 답은 첫 문장부터 바로 읽습니다.
    - 한 번 대답한 뒤 3초 동안은 다시 부르지 않은 것으로 봅니다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

# pygame 이 콘솔에 홍보 문구를 찍지 않게 한다
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import numpy as np
except Exception:  # numpy 가 없으면 --check 가 알려 준다
    np = None  # type: ignore[assignment]

APP_NAME = "자비스"
APP_VERSION = "4.0.0"
APP_TITLE = "자비스 (음성 비서)"

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
EXAMPLE_CONFIG_NAME = "config.example.json"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")

SAMPLE_RATE = 16000          # Vosk 와 받아쓰기가 함께 쓰는 샘플레이트
BLOCK_MS = 30                # 한 번에 읽는 소리 길이

# 종료 코드: 정상 종료(사용자가 껐음). 실행기(.vbs)가 이것을 보면 다시 켜지 않는다.
EXIT_USER_QUIT = 78

# 한국어 소형 모델 (약 87MB, 풀면 약 250MB)
VOSK_MODEL_NAME = "vosk-model-small-ko-0.22"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip"
VOSK_MODEL_MB = 87
MIN_ZIP_BYTES = 20 * 1024 * 1024   # 이보다 작으면 내려받기가 제대로 안 된 것으로 본다

DEFAULT_CONFIG = {
    # ── 부르는 말 ──────────────────────────────────────────────
    "vosk_model_path": "",
    "wake_words": ["자비스", "헤이 자비스"],
    "wake_engine": "hybrid",        # hybrid(권장) | fast(Vosk만) | accurate(Whisper만)
    "wake_sensitivity": 0.55,
    "wake_reply": "네, 대표님",
    "wake_cooldown_sec": 3.0,       # 한 번 대답한 뒤 이만큼은 다시 부르지 않은 것으로 본다
    # ── 받아쓰기 ──────────────────────────────────────────────
    "whisper_model": "small",       # tiny | base | small | medium
    # ── 답변 ──────────────────────────────────────────────────
    "max_answer_chars": 400,
    "stream_answer": True,          # 첫 문장부터 바로 읽기
    "claude_exe": "claude",
    "claude_timeout_sec": 180,
    "claude_cwd": "",
    "claude_model": "",             # 비우면 클로드 코드 기본값
    # ── 목소리 ────────────────────────────────────────────────
    "tts_voice": "ko-KR-SunHiNeural",
    "tts_engine": "auto",           # auto | edge | windows
    # ── 마이크 ────────────────────────────────────────────────
    "mic_device": "",               # 비우면 윈도우 기본 마이크. 이름 일부 또는 번호
    "mic_gain": 1.0,                # 마이크가 너무 작게 들리면 2.0~4.0
    "silence_sec": 0.8,             # 이만큼 조용하면 말이 끝난 것으로 본다
    "max_record_sec": 12.0,
    "min_record_sec": 0.8,
    "listen_wait_sec": 6.0,         # 부른 뒤 질문을 기다려 주는 시간
    "speech_threshold": 0.008,      # 말소리로 볼 최소 크기(0~1). 시끄러우면 0.02~0.05
    # ── 화면 · 기타 ───────────────────────────────────────────
    "show_window": True,
    "owner_name": "김용진",
    "assistant_name": "자비스",
    "announce_start": False,
    "log_keep_days": 14,
}

# v1(Picovoice 판)·v3 에서 쓰던 항목. 남아 있으면 지워도 된다고 알려 준다.
LEGACY_CONFIG_KEYS = (
    "picovoice_access_key",
    "wake_word_path",
    "wake_word_model_path",
    "wake_word_sensitivity",
    "wake_phrase",
    "fallback_wake_phrase",
)

log = logging.getLogger("davis")


# ---------------------------------------------------------------------------
# 작은 도우미들
# ---------------------------------------------------------------------------

def _p(*args, **kwargs):
    """pythonw(창 없음)에서도 죽지 않는 print."""
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except Exception:
        pass


def _need_numpy() -> Any:
    if np is None:
        raise RuntimeError("numpy 를 불러오지 못했습니다. install.bat 을 다시 실행해 주세요.")
    return np


_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\u2B00-\u2BFF]")


def clean_for_speech(text: str) -> str:
    """읽어 줄 문장에서 마크다운·이모지·주소를 걷어 낸다."""
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", t)      # [글자](주소) → 글자
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)
    t = t.replace("`", "").replace("*", "").replace("#", "").replace("_", "")
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def trim_answer(text: str, max_chars: int) -> str:
    """답변이 너무 길면 문장 단위로 자른다."""
    text = (text or "").strip()
    max_chars = int(max_chars or 400)
    if len(text) <= max_chars:
        return text
    parts = re.split(r"(?<=[.!?。？！])\s+", text)
    out = ""
    for part in parts:
        candidate = (out + " " + part).strip()
        if len(candidate) > max_chars:
            break
        out = candidate
    if not out:
        out = text[:max_chars].rstrip()
        if out and out[-1] not in ".!?。？！":
            out += "…"
    return out


def split_sentences(buffer: str, min_chars: int = 12) -> tuple[list[str], str]:
    """스트리밍 답변을 문장 단위로 잘라 낸다. (읽을 문장들, 남은 글자)

    너무 짧은 토막('네.')은 다음 문장과 붙여서 읽는다.
    """
    out: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?。？！]+|\n+", buffer):
        if m.end() - start >= max(1, int(min_chars)):
            piece = buffer[start:m.end()].strip()
            if piece:
                out.append(piece)
            start = m.end()
    return out, buffer[start:]


def _local_app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / "davis-voice"
    return Path.home() / ".davis-voice"


def _can_write(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def writable_dir(candidates: list[Path]) -> Path:
    """쓸 수 있는 첫 폴더를 돌려준다. (프로그램 폴더 → 사용자 폴더 순)"""
    for folder in candidates:
        if _can_write(folder):
            return folder
    return _local_app_dir()


def log_dir() -> Path:
    """로그는 프로그램 폴더의 logs\\ 에 쓰고, 안 되면 사용자 폴더에 쓴다."""
    return writable_dir([BASE_DIR / "logs", _local_app_dir() / "logs"])


def cache_dir() -> Path:
    d = _local_app_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 로그 — 날짜별 파일 + 화면
# ---------------------------------------------------------------------------

class DailyFileHandler(logging.Handler):
    """logs\\YYYY-MM-DD.log 에 하루 단위로 남긴다. 자정이 지나면 새 파일로 넘어간다."""

    def __init__(self, folder: Path, keep_days: int = 14) -> None:
        super().__init__()
        self.folder = folder
        self.keep_days = int(keep_days or 14)
        self._day: str = ""
        self._fp = None

    def _open_for(self, day: str) -> None:
        try:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:
                    pass
            self.folder.mkdir(parents=True, exist_ok=True)
            self._fp = (self.folder / f"{day}.log").open("a", encoding="utf-8")
            self._day = day
        except Exception:
            self._fp = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = date.today().isoformat()
            if day != self._day or self._fp is None:
                self._open_for(day)
            if self._fp is None:
                return
            self._fp.write(self.format(record) + "\n")
            self._fp.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._fp is not None:
                self._fp.close()
        except Exception:
            pass
        self._fp = None
        super().close()


def _prune_logs(folder: Path, keep_days: int) -> None:
    try:
        files = sorted(folder.glob("????-??-??.log"))
        for old in files[: max(0, len(files) - int(keep_days))]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass


def setup_logging(console: bool = False, keep_days: int = 14) -> logging.Logger:
    logger = logging.getLogger("davis")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    folder = log_dir()
    _prune_logs(folder, keep_days)
    fh = DailyFileHandler(folder, keep_days)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if console and sys.stdout is not None:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def _install_global_handlers(logger: logging.Logger, on_fatal: Callable[[str], None] | None = None) -> None:
    """어디서 터진 오류든 로그에 남긴다. (조용히 죽지 않게)"""
    def hook(exc_type, exc_value, tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, tb))
        try:
            logger.critical("처리되지 않은 오류로 멈췄습니다:\n%s", text)
        except Exception:
            pass
        if on_fatal is not None:
            try:
                on_fatal(f"{exc_type.__name__}: {exc_value}")
            except Exception:
                pass

    sys.excepthook = hook
    try:
        threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)
    except Exception:
        pass
    # 윈도우에서 파이썬이 통째로 죽는 경우(C DLL 충돌 등)의 흔적
    try:
        crash = log_dir() / "crash.log"
        import faulthandler
        faulthandler.enable(file=open(crash, "a", encoding="utf-8"), all_threads=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["wake_words"] = list(DEFAULT_CONFIG["wake_words"])
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as fp:
                user = json.load(fp)
            if isinstance(user, dict):
                unknown = [k for k in user if k not in DEFAULT_CONFIG]
                for key, value in user.items():
                    if key in DEFAULT_CONFIG and value is not None:
                        cfg[key] = value
                if unknown:
                    legacy = [k for k in unknown if k in LEGACY_CONFIG_KEYS]
                    if legacy:
                        log.info("예전 설정 항목은 무시합니다: %s", ", ".join(legacy))
                    other = [k for k in unknown if k not in LEGACY_CONFIG_KEYS]
                    if other:
                        log.warning("알 수 없는 설정 항목을 무시합니다: %s", ", ".join(other))
        except Exception as exc:
            log.warning("config.json 을 읽지 못했습니다: %s (기본값으로 실행합니다)", exc)
    if not isinstance(cfg.get("wake_words"), list) or not cfg["wake_words"]:
        cfg["wake_words"] = list(DEFAULT_CONFIG["wake_words"])
    return cfg


def save_config(cfg: dict) -> None:
    data = {k: cfg.get(k, DEFAULT_CONFIG.get(k)) for k in DEFAULT_CONFIG}
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def _expand(path_text: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path_text).strip().strip('"')))).resolve()


# ---------------------------------------------------------------------------
# 한국어 음성모델(Vosk) — 찾기 · 내려받기 · 풀기
# ---------------------------------------------------------------------------

class ModelError(Exception):
    pass


def is_model_dir(path: Path | str | None) -> bool:
    """Vosk 모델 폴더인지 확인한다."""
    if not path:
        return False
    p = Path(path)
    try:
        return (
            p.is_dir()
            and (p / "am" / "final.mdl").is_file()
            and (p / "conf" / "model.conf").is_file()
        )
    except OSError:
        return False


def model_search_dirs(cfg: dict) -> list[Path]:
    dirs: list[Path] = []
    text = str(cfg.get("vosk_model_path") or "").strip()
    if text:
        dirs.append(_expand(text))
    dirs.append(BASE_DIR / VOSK_MODEL_NAME)
    dirs.append(BASE_DIR)
    dirs.append(_local_app_dir() / "models" / VOSK_MODEL_NAME)
    dirs.append(_local_app_dir() / "models")
    return dirs


def find_model_dir(cfg: dict) -> Path | None:
    """설정값 → 프로그램 폴더 → 사용자 폴더 순으로 모델을 찾는다."""
    for folder in model_search_dirs(cfg):
        if is_model_dir(folder):
            return folder
    for folder in (BASE_DIR, _local_app_dir() / "models"):
        try:
            for child in sorted(folder.glob("vosk-model*")):
                if is_model_dir(child):
                    return child
        except OSError:
            continue
    return None


def _model_install_parent() -> Path:
    """모델을 풀어 둘 폴더. 프로그램 폴더에 쓸 수 없으면 사용자 폴더를 쓴다."""
    if _can_write(BASE_DIR):
        return BASE_DIR
    parent = _local_app_dir() / "models"
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _download_zip(url: str, zip_path: Path, on_progress=None,
                  min_bytes: int = MIN_ZIP_BYTES) -> Path:
    """모델 압축 파일을 내려받는다. 이미 받아 둔 파일이 멀쩡하면 그대로 쓴다."""
    import urllib.request

    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path

    partial = zip_path.with_suffix(zip_path.suffix + ".part")
    try:
        if partial.exists():
            partial.unlink()
    except Exception:
        pass

    request = urllib.request.Request(url, headers={"User-Agent": "davis-installer/4.0"})
    done = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with partial.open("wb") as fp:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                fp.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)
                if total and done >= total:
                    break

    if done < int(min_bytes):
        raise ModelError(f"내려받은 파일이 너무 작습니다({done / 1024 / 1024:.1f}MB).")
    if not zipfile.is_zipfile(partial):
        raise ModelError("내려받은 파일이 압축 파일이 아닙니다.")
    partial.replace(zip_path)
    return zip_path


def _safe_extract(zip_path: Path, dest_parent: Path) -> Path:
    """압축을 임시 폴더에 풀고, 모델 폴더만 제자리로 옮긴다."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="davis_model_", dir=str(dest_parent)))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            broken = zf.testzip()
            if broken:
                raise ModelError(f"압축 파일이 손상되었습니다({broken}).")
            staging_root = staging.resolve()
            for member in zf.infolist():
                target = (staging / member.filename).resolve()
                if not str(target).startswith(str(staging_root)):
                    raise ModelError("압축 파일 안에 이상한 경로가 있어 풀지 않았습니다.")
            zf.extractall(staging)

        inner = staging / VOSK_MODEL_NAME
        if not is_model_dir(inner):
            found = [p for p in staging.iterdir() if p.is_dir() and is_model_dir(p)]
            if not found:
                raise ModelError("압축을 풀었지만 모델 폴더를 찾지 못했습니다.")
            inner = found[0]

        dest = dest_parent / VOSK_MODEL_NAME
        if dest.exists():
            backup = dest.with_name(f"{dest.name}.old-{int(time.time())}")
            try:
                dest.rename(backup)
            except Exception:
                pass
        inner.rename(dest)
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_model(cfg: dict, logger: logging.Logger = log, on_progress=None) -> Path:
    """모델을 찾고, 없으면 내려받아 푼다. 성공하면 모델 폴더 경로를 돌려준다."""
    found = find_model_dir(cfg)
    if found is not None:
        if str(cfg.get("vosk_model_path") or "").strip() != str(found):
            cfg["vosk_model_path"] = str(found)
            try:
                save_config(cfg)
            except Exception:
                pass
        return found

    parent = _model_install_parent()
    zip_path = cache_dir() / f"{VOSK_MODEL_NAME}.zip"
    logger.info("한국어 음성모델을 내려받습니다: %s", VOSK_MODEL_URL)
    zip_path = _download_zip(VOSK_MODEL_URL, zip_path, on_progress=on_progress)
    dest = _safe_extract(zip_path, parent)
    if not is_model_dir(dest):
        raise ModelError("모델을 풀었지만 확인하지 못했습니다.")
    cfg["vosk_model_path"] = str(dest)
    try:
        save_config(cfg)
    except Exception:
        pass
    logger.info("한국어 음성모델 준비 완료: %s", dest)
    return dest


def model_size_mb(path: Path) -> float:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        pass
    return total / 1024 / 1024


_VOSK_READY = False


def _import_vosk(logger: logging.Logger = log):
    global _VOSK_READY
    try:
        import vosk
    except Exception as exc:
        raise ModelError(f"vosk 를 불러오지 못했습니다: {exc}")
    if not _VOSK_READY:
        try:
            vosk.SetLogLevel(-1)   # Kaldi 로그가 파일에 쏟아지지 않게
        except Exception:
            pass
        _VOSK_READY = True
    return vosk


class VoskHolder:
    """Vosk 모델은 한 번만 읽어서 계속 쓴다(읽는 데 몇 초 걸린다)."""

    def __init__(self) -> None:
        self.model = None
        self.path: Path | None = None
        self.error: Exception | None = None
        self._lock = threading.Lock()

    def get(self, cfg: dict, logger: logging.Logger = log):
        with self._lock:
            if self.model is not None:
                return self.model
            vosk = _import_vosk(logger)
            path = find_model_dir(cfg)
            if path is None:
                raise ModelError(
                    "한국어 음성모델이 없습니다. install.bat 을 다시 실행하거나 "
                    "'davis.py --setup' 을 실행해 주세요."
                )
            logger.info("한국어 음성모델을 읽는 중입니다: %s", path)
            started = time.time()
            try:
                self.model = vosk.Model(str(path))
            except Exception as exc:
                raise ModelError(f"한국어 음성모델을 읽지 못했습니다: {exc}")
            self.path = path
            logger.info("음성모델 준비 완료 (%.1f초)", time.time() - started)
            return self.model

    def peek(self):
        return self.model


# ---------------------------------------------------------------------------
# 부르는 말 — 문법 · 문자열 판정
# ---------------------------------------------------------------------------

# Vosk 가 '자비스'를 흐리게 들을 때 내놓는 다른 표기들 (이 모델 어휘에 있는 단어만)
WAKE_STRONG_EXTRA = ("자비스",)

# 어휘에 없어서 문법에는 넣을 수 없지만, 들리면 깨운 것으로 보는 표기들
WAKE_MATCH_ONLY = ("자비수", "자비쓰", "자빗스", "짜비스")

# Whisper 가 들려주는 표기들. 실제 측정에서 '헤이 자비스'가 '헤이 잡이스'로,
# 시끄러운 곳에서는 '자비스'가 '자비스'로 정확히 나왔다. ('자비를' 같은 말은 넣지 않는다)
WAKE_WHISPER_FORMS = {
    "자비스", "잡이스", "자비쓰", "자비수", "자빗스", "짜비스", "쟈비스", "자비쓰",
    "자비스야", "자비스요", "자비쓰야",
}

# 앞 글자를 놓쳤을 때 나오는 형태. 말 전체가 아주 짧을 때만 인정한다.
WAKE_WEAK = ("비스",)

# 깨우기 말로 인정하려면 그 말이 실제로 소리 난 길이가 이만큼은 되어야 한다.
# (측정: 실제 '자비스'는 0.66~0.93초, 다른 말이 우겨넣어진 경우는 0.3초 안팎)
MIN_WAKE_SEC = 0.45

# 문법에 함께 넣어 두는 흔한 말들(Vosk 1차 확인용).
# 이 말들이 있어야 Vosk 가 다른 말을 '자비스'로 우겨넣지 않는다.
DISTRACTOR_WORDS = (
    "비스", "비스킷", "비스듬", "자비", "서비스", "디스", "디스크",
    "티켓", "로션", "베스트", "버스",
    "안녕", "고마워", "미안", "괜찮아", "좋아", "싫어", "맞아", "아니", "네", "예",
    "아니요", "그래", "그래요", "옳지", "네네", "아하", "잠깐", "잠시", "감사", "죄송",
    "뭐", "뭘", "뭐야", "뭐해", "왜", "왜요", "누구", "언제", "어디", "얼마", "몇", "며칠",
    "무슨", "어떤", "어떻게", "어때", "그거", "이거", "저거", "그것", "이것", "저것", "저기",
    "나", "너", "우리", "저", "제", "내", "님", "사람", "사람들", "아들", "딸", "엄마", "아빠",
    "부모님", "친구", "선배", "후배", "대표", "사장", "회사", "사무실", "직원", "손님", "고객",
    "대표님", "사장님", "고객님", "팀장님", "직원분", "회사원", "거래처", "영업팀",
    "오늘", "어제", "내일", "요즘", "아침", "점심", "저녁", "밤", "새벽", "오전", "오후",
    "시간", "시", "분", "일찍", "지금", "먼저", "다음", "이전", "이번", "지난", "다음주",
    "이번주", "한달", "일년", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
    "약속", "회의", "회계", "자료", "보고서", "계약", "매출", "일정", "계획", "할일", "목록",
    "정리", "요약", "번역", "계산", "확인", "점검", "준비", "진행", "상태", "결과", "문제",
    "해결", "방법", "이유", "내용", "질문", "대답", "답변", "설명", "문서", "메일", "뉴스",
    "광고", "마케팅", "홍보", "노출", "순위", "키워드", "플레이스", "네이버", "블로그", "인스타그램",
    "유튜브", "카톡", "문자메시지",
    "집", "차", "자동차", "택시", "지하철", "역", "버스", "학교", "병원", "은행", "마트", "편의점",
    "식당", "카페", "주차장", "공원", "밥", "물", "커피", "술", "맥주", "소주", "김치", "라면", "빵",
    "사과", "바나나", "음식", "한잔", "전화", "문자", "음악", "노래", "영상", "영화", "드라마", "사진",
    "파일", "폴더", "컴퓨터", "노트북", "핸드폰", "화면", "볼륨", "소리", "목소리", "알람", "예약",
    "취소", "배송", "주문", "결제", "가격", "할인", "쿠폰", "상품", "날씨", "비", "눈", "바람",
    "춥다", "미세먼지",
    "해", "해요", "했어", "합니다", "줘", "줄래", "줄게", "줄까", "가", "갈", "갈게", "갈래", "갔어",
    "와", "왔어", "봐", "봤어", "볼게", "볼래", "할게", "할래", "먹을", "먹어", "먹을까", "앉아",
    "있어", "있지", "있어요", "없어", "없다", "돼", "됐어", "알려", "알려줘", "보여", "보여줘",
    "틀어", "켜", "꺼", "정지", "시작", "시작해", "종료", "멈춰", "검색", "찾아", "찾아봐", "열어",
    "닫아", "보내", "읽어", "읽어줘", "써", "써줘", "말해", "말해줘", "물어", "물어봐", "크게",
    "작게", "베풀어", "좋았어", "참", "좋은", "나쁜", "많이", "조금", "아주", "정말", "진짜", "빨리",
    "천천히", "다시", "한번", "두번", "좀", "잘", "못", "안", "더", "덜", "그리고", "그런데",
    "하지만", "그래서", "은",
)


def squash(text: str) -> str:
    """띄어쓰기·문장부호를 지우고 비교하기 좋게 만든다."""
    return re.sub(r"[\s\W_]+", "", str(text or "")).lower()


def wake_words(cfg: dict) -> list[str]:
    words = []
    configured = cfg.get("wake_words") or []
    if isinstance(configured, str):
        configured = [configured]
    for item in list(configured) + list(WAKE_STRONG_EXTRA) + list(WAKE_MATCH_ONLY):
        text = str(item or "").strip()
        if text and squash(text) not in [squash(w) for w in words]:
            words.append(text)
    return words


def wake_grammar(cfg: dict) -> str:
    """Vosk 에 넘길 문법(인식할 단어 목록)."""
    words: list[str] = []
    seen: set[str] = set()
    configured = cfg.get("wake_words") or []
    if isinstance(configured, str):
        configured = [configured]
    source = list(configured) + list(WAKE_STRONG_EXTRA) + list(WAKE_WEAK) + list(DISTRACTOR_WORDS)
    for item in source:
        text = str(item or "").strip()
        key = squash(text)
        if text and key and key not in seen:
            seen.add(key)
            words.append(text)
    return json.dumps(words, ensure_ascii=False)


def _sensitivity(cfg: dict) -> float:
    try:
        value = float(cfg.get("wake_sensitivity"))
    except (TypeError, ValueError):
        value = float(DEFAULT_CONFIG["wake_sensitivity"])
    return min(1.0, max(0.0, value))


def _char_distance(a: str, b: str) -> int:
    """두 글자열이 몇 글자나 다른지(레벤슈타인)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _token_list(words: list | None) -> list[tuple[str, float, float, str]]:
    """Vosk 가 준 단어 목록을 (비교용 글자, 시작, 끝, 원래 글자)로 바꾼다."""
    tokens: list[tuple[str, float, float, str]] = []
    for item in words or []:
        try:
            tokens.append((
                squash(item.get("word", "")),
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
                str(item.get("word", "")),
            ))
        except (TypeError, ValueError, AttributeError):
            continue
    return tokens


def _match_tokens(tokens: list[tuple[str, float, float, str]], forms: set[str],
                  min_sec: float) -> str | None:
    """단어별 시간이 있을 때: 깨우기 말이 '한 단어로, 충분히 길게' 들렸는지 본다."""
    for key, start, end, raw in tokens:
        if key and key in forms and (end - start) >= min_sec:
            return raw or key
    for i in range(len(tokens) - 1):
        joined = tokens[i][0] + tokens[i + 1][0]
        if joined and joined in forms and (tokens[i + 1][2] - tokens[i][1]) >= min_sec:
            return tokens[i + 1][3] or joined
    return None


def find_wake(text: str, cfg: dict, speech_sec: float | None = None,
              sensitivity: float | None = None, words: list | None = None) -> str | None:
    """Vosk 가 들은 말에서 깨우기 말을 찾는다. 찾으면 그 말을 돌려준다."""
    heard = squash(text)
    if not heard:
        return None
    level = _sensitivity(cfg) if sensitivity is None else float(sensitivity)
    configured = wake_words(cfg)
    forms = {squash(w) for w in configured if squash(w)}
    tokens = _token_list(words) if words else []

    # 1) '자비스' 계열이 그대로 들린 경우 — 문장 중간에 있어도 인정
    if tokens:
        hit = _match_tokens(tokens, forms, MIN_WAKE_SEC)
        if hit:
            return hit
    else:
        for word in configured:
            key = squash(word)
            if key and key in heard:
                return word

    weak_keys = {squash(w) for w in WAKE_WEAK}

    # 2) 한 글자를 놓친 짧은 말 — 그 말 전체가 그 형태일 때만
    if level >= 0.3 and speech_sec is not None and speech_sec <= 1.6:
        if heard in weak_keys:
            return heard

    # 3) '비스, 오늘 날씨 어때' 처럼 부르면서 이어 말한 경우 — 첫 단어만 본다
    if level >= 0.3 and tokens:
        key, start, end, raw = tokens[0]
        if key in weak_keys and (end - start) <= 1.0:
            return raw or key

    # 4) 아주 느슨하게: 세 글자가 한 글자만 달라도
    primary = squash(configured[0]) if configured else ""
    if level >= 0.85 and speech_sec is not None and speech_sec <= 1.6 and len(heard) == 3 and len(primary) == 3:
        if _char_distance(heard, primary) <= 1:
            return heard
    return None


# 깨우기 말 뒤에 붙는 조사 — 지워도 뜻이 남는다
_TRAILING_PARTICLES = ("이랑", "랑", "를", "을", "이", "가", "은", "는", "도", "만", "야", "아", "요", "좀", "!")


def whisper_wake_hit(text: str, cfg: dict) -> str | None:
    """Whisper 가 받아쓴 문장에서 깨우기 말을 찾는다.

    실제 측정에서 Whisper 는 시끄러운 곳에서도 '자비스'를 정확히 받아썼고,
    '자비를 베풀어 주세요'·'비스킷'·'서비스'는 깨우기 말로 오인하지 않았다.
    그래서 여기서는 '비슷한 소리'를 넓게 받아 주지 않고, 정해진 표기만 인정한다.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    flat = re.sub(r"[^\w가-힣\s]", " ", raw).lower()
    if re.search(r"\b(hey\s*jarvis|hi\s*jarvis|자비스|잡이스)\b", flat):
        return "자비스"
    tokens = [t for t in flat.split() if t]
    configured = {squash(w) for w in (cfg.get("wake_words") or []) if squash(w)}
    joined_all = ""
    for token in tokens:
        base = token
        for particle in _TRAILING_PARTICLES:
            if len(base) > len(particle) and base.endswith(particle):
                base = base[: -len(particle)]
                break
        joined_all += base
        if base in WAKE_WHISPER_FORMS or (base and squash(base) in configured):
            return base
    # '자 비스' 처럼 한 단어가 두 토큰으로 갈라진 경우
    if "자비스" in squash(" ".join(tokens)) or "잡이스" in squash(" ".join(tokens)):
        return "자비스"
    return None


def strip_wake_prefix(text: str, cfg: dict) -> str:
    """받아쓴 문장 앞에 붙은 깨우기 말('자비스')을 지운다."""
    s = str(text or "").strip()
    if not s:
        return ""
    forms = {squash(w) for w in wake_words(cfg)} | {squash(w) for w in WAKE_WEAK} | WAKE_WHISPER_FORMS
    forms = {f for f in forms if f}
    tokens = s.split()
    for count in (3, 2, 1):
        if count > len(tokens):
            continue
        head = " ".join(tokens[:count])
        key = squash(head)
        if key in forms or (count == 3 and key in ("헤이자비스", "헤이잡이스")):
            return " ".join(tokens[count:]).strip(" ,.!?~")
    # 깨우기 말 뒤에 조사가 붙은 경우 (자비스를, 자비스야)
    if tokens:
        first = tokens[0]
        base = first
        for particle in _TRAILING_PARTICLES:
            if len(base) > len(particle) and base.endswith(particle):
                base = base[: -len(particle)]
                break
        if squash(base) in forms:
            return " ".join(tokens[1:]).strip(" ,.!?~")
    # '헤이 자비스'가 '헤이잡이스'로 붙어 나온 경우
    squashed = squash(s)
    for form in sorted(forms, key=len, reverse=True):
        if len(form) >= 3 and squashed.startswith(form):
            return s[len(form):].strip(" ,.!?~") if len(s) > len(form) else ""
    return s


# ---------------------------------------------------------------------------
# 오디오 장치
# ---------------------------------------------------------------------------

class MicError(Exception):
    pass


def _sounddevice():
    try:
        import sounddevice as sd
        return sd
    except Exception as exc:
        raise MicError(f"오디오 라이브러리(sounddevice)를 불러오지 못했습니다: {exc}")


def list_input_devices() -> list[dict]:
    """쓸 수 있는 마이크 목록. ([{index, name, channels, rate}, ...])"""
    sd = _sounddevice()
    out: list[dict] = []
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise MicError(f"오디오 장치 목록을 읽지 못했습니다: {exc}")
    for index, dev in enumerate(devices):
        try:
            if int(dev.get("max_input_channels", 0)) > 0:
                out.append({
                    "index": index,
                    "name": str(dev.get("name", "")),
                    "channels": int(dev.get("max_input_channels", 0)),
                    "rate": int(float(dev.get("default_samplerate", 0) or 0)),
                })
        except Exception:
            continue
    return out


_VIRTUAL_MIC_HINTS = ("stereo mix", "스테레오 믹스", "what u hear", "wave out", "vb-audio",
                      "virtual", "가상", "cable output", "loopback")


def pick_input_device(cfg: dict, logger: logging.Logger = log) -> tuple[int | None, str, list[dict]]:
    """쓸 마이크를 고른다. (장치 번호 또는 None, 이름, 전체 목록)"""
    devices = list_input_devices()
    if not devices:
        raise MicError("마이크(입력 장치)를 하나도 찾지 못했습니다. 마이크가 꽂혀 있는지 확인해 주세요.")
    want = str(cfg.get("mic_device") or "").strip()
    if want:
        if want.isdigit():
            index = int(want)
            for dev in devices:
                if dev["index"] == index:
                    return index, dev["name"], devices
            raise MicError(f"mic_device 에 적힌 {index}번 장치는 마이크가 아닙니다. --list-mics 로 확인해 주세요.")
        low = want.lower()
        for dev in devices:
            if low in dev["name"].lower():
                return dev["index"], dev["name"], devices
        logger.warning("mic_device 에 적힌 이름(%s)을 찾지 못해 기본 마이크를 씁니다.", want)

    sd = _sounddevice()
    default_index = None
    default_name = ""
    try:
        default_index = int(sd.default.device[0]) if sd.default.device else None
    except Exception:
        default_index = None
    try:
        info = sd.query_devices(kind="input")
        default_name = str(info.get("name", ""))
        if default_index is None:
            for dev in devices:
                if dev["name"] == default_name:
                    default_index = dev["index"]
                    break
    except Exception:
        pass

    if default_index is not None and any(d["index"] == default_index for d in devices):
        # '스테레오 믹스' 같은 가상 장치가 기본이면 진짜 마이크를 찾아 본다
        if not any(hint in default_name.lower() for hint in _VIRTUAL_MIC_HINTS):
            return default_index, default_name or f"{default_index}번", devices
        for dev in devices:
            if not any(hint in dev["name"].lower() for hint in _VIRTUAL_MIC_HINTS):
                logger.warning("기본 입력 장치가 '%s'(가상 장치)여서 '%s'를 대신 씁니다.",
                               default_name, dev["name"])
                return dev["index"], dev["name"], devices
        return default_index, default_name, devices
    return None, default_name or "(윈도우 기본)", devices


def devices_report(cfg: dict) -> str:
    """마이크 목록을 한글 안내문으로 만든다."""
    lines = []
    try:
        devices = list_input_devices()
    except MicError as exc:
        return str(exc)
    index, name, _ = (None, "", [])
    try:
        index, name, _ = pick_input_device(cfg)
    except Exception:
        pass
    lines.append(f"쓰는 마이크: {name}" + (f" ({index}번)" if index is not None else ""))
    lines.append("마이크 목록:")
    for dev in devices:
        mark = " ← 지금 씀" if index is not None and dev["index"] == index else ""
        lines.append(f"  [{dev['index']}] {dev['name']} (입력 {dev['channels']}채널, 기본 {dev['rate']}Hz){mark}")
    return "\n".join(lines)


class Resampler:
    """마이크 샘플레이트를 16kHz 로 바꾼다.

    예전 판은 선형 보간만 써서 48kHz 마이크의 높은 소리가 낮은 쪽으로 접혀 들어왔다
    (에일리어싱). 그러면 '자비스'의 ㅅ·ㅈ 소리가 뭉개져 Vosk 가 '서비스'로 잘못 듣는다.
    여기서는 먼저 부드럽게 걸러 낸(저역통과) 뒤 줄인다.
    """

    def __init__(self, from_rate: int) -> None:
        self.from_rate = int(from_rate or SAMPLE_RATE)
        self._taps: Any = None
        self._state: Any = None
        if self.from_rate != SAMPLE_RATE and np is not None:
            self._taps = self._design_lowpass()

    def _design_lowpass(self) -> Any:
        """7.2kHz 언저리에서 자르는 창 함수(sinc) 필터."""
        taps = 63
        cutoff = 7200.0 / self.from_rate        # 0~0.5 정규화 주파수
        n = np.arange(taps) - (taps - 1) / 2.0
        h = 2 * cutoff * np.sinc(2 * cutoff * n)
        h *= np.hamming(taps)
        h /= np.sum(h)
        self._state = np.zeros(taps - 1, dtype=np.float32)
        return h.astype(np.float32)

    def __call__(self, pcm: Any) -> Any:
        np_mod = _need_numpy()
        x = np_mod.asarray(pcm, dtype=np_mod.int16).reshape(-1)
        if self.from_rate == SAMPLE_RATE or x.size == 0:
            return x
        f = x.astype(np_mod.float32)
        if self._taps is not None:
            padded = np_mod.concatenate([self._state, f])
            filtered = np_mod.convolve(padded, self._taps, mode="valid")
            self._state = padded[-(len(self._taps) - 1):]
            f = filtered
        n = int(round(f.size * SAMPLE_RATE / float(self.from_rate)))
        if n <= 0:
            return np_mod.zeros(0, dtype=np_mod.int16)
        idx = np_mod.linspace(0, f.size - 1, n)
        out = np_mod.interp(idx, np_mod.arange(f.size), f)
        return np_mod.clip(np_mod.round(out), -32768, 32767).astype(np_mod.int16)


class MicStream:
    """16kHz int16 덩어리를 계속 내놓는 마이크 스트림.

    - 장치를 직접 고를 수 있다(mic_device).
    - 16kHz 가 안 되면 그 장치의 기본 레이트로 열고, 걸러서 16kHz 로 바꾼다.
    - 열 수 없으면 쓸 수 있는 마이크 목록을 한글로 알려 준다.
    """

    def __init__(self, cfg: dict, logger: logging.Logger = log) -> None:
        self.cfg = cfg
        self.logger = logger
        self.stream = None
        self.device_index: int | None = None
        self.device_name = ""
        self.rate = SAMPLE_RATE
        self.resampler = Resampler(SAMPLE_RATE)
        self.gain = 1.0
        try:
            self.gain = max(0.1, min(10.0, float(cfg.get("mic_gain") or 1.0)))
        except (TypeError, ValueError):
            self.gain = 1.0
        self.blocksize = int(SAMPLE_RATE * BLOCK_MS / 1000)

    def open(self) -> "MicStream":
        _need_numpy()
        sd = _sounddevice()
        index, name, devices = pick_input_device(self.cfg, self.logger)
        self.device_index, self.device_name = index, name
        errors: list[str] = []

        rates: list[int] = []
        for candidate in (SAMPLE_RATE, 48000, 44100):
            if candidate not in rates:
                rates.append(candidate)
        if index is not None:
            try:
                info = sd.query_devices(index)
                dev_rate = int(float(info.get("default_samplerate") or 0))
                if dev_rate and dev_rate not in rates:
                    rates.append(dev_rate)
            except Exception:
                pass

        for rate in rates:
            try:
                stream = sd.RawInputStream(
                    samplerate=rate,
                    blocksize=int(rate * BLOCK_MS / 1000),
                    dtype="int16",
                    channels=1,
                    device=index,
                )
                stream.start()
                self.stream = stream
                self.rate = rate
                self.resampler = Resampler(rate)
                self.logger.info("마이크를 열었습니다: %s (%d번, %dHz)", name, index if index is not None else -1, rate)
                return self
            except Exception as exc:
                errors.append(f"{rate}Hz: {exc}")
                self.logger.warning("마이크를 %dHz 로 열지 못했습니다: %s", rate, exc)

        detail = "\n".join(errors[-3:])
        raise MicError(
            "마이크를 열지 못했습니다.\n"
            f"고른 마이크: {name}\n"
            f"마지막 오류: {detail}\n"
            "윈도우 설정 → 개인정보 및 보안 → 마이크 에서 '앱이 마이크에 액세스'가 켜져 있는지,"
            " 다른 프로그램(줌·녹음기)이 마이크를 쓰고 있지 않은지 확인해 주세요.\n"
            + devices_report(self.cfg)
        )

    def read_block(self) -> Any:
        """30ms(16kHz) 분량 int16 배열. 실패하면 MicError."""
        if self.stream is None:
            raise MicError("마이크가 열려 있지 않습니다.")
        np_mod = _need_numpy()
        frames = int(self.rate * BLOCK_MS / 1000)
        try:
            data, overflow = self.stream.read(frames)
        except Exception as exc:
            raise MicError(f"마이크에서 소리를 읽지 못했습니다: {exc}")
        pcm = np_mod.frombuffer(data, dtype=np_mod.int16)
        if self.rate != SAMPLE_RATE:
            pcm = self.resampler(pcm)
        if self.gain != 1.0:
            pcm = np_mod.clip(pcm.astype(np_mod.float32) * self.gain, -32768, 32767).astype(np_mod.int16)
        return pcm

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


def _rms(pcm) -> float:
    np_mod = _need_numpy()
    x = np_mod.asarray(pcm, dtype=np_mod.float32) / 32768.0
    if x.size == 0:
        return 0.0
    return float(np_mod.sqrt(float(np_mod.mean(np_mod.square(x)))))


class MicReader:
    """마이크를 읽는 일만 하는 스레드. 읽은 소리는 큐에 쌓인다.

    받아쓰기(Whisper)는 몇백 ms 걸린다. 그동안 마이크를 못 읽으면 질문 앞부분을
    놓치므로, 읽기와 처리를 나눈다.
    """

    def __init__(self, stream: MicStream, logger: logging.Logger = log, maxlen: int = 400) -> None:
        self.stream = stream
        self.logger = logger
        self.queue: queue.Queue = queue.Queue(maxsize=maxlen)
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mic", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pcm = self.stream.read_block()
            except Exception as exc:
                self.error = exc
                self.logger.error("마이크 읽기 중단: %s", exc)
                return
            try:
                self.queue.put_nowait(pcm)
            except queue.Full:
                try:
                    self.queue.get_nowait()   # 가장 오래된 것부터 버린다
                    self.queue.put_nowait(pcm)
                    self.dropped += 1
                except Exception:
                    pass

    def get(self, timeout: float = 0.5):
        """블록 하나. 시간이 지나면 None."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, seconds: float = 0.0) -> None:
        """쌓인 소리를 버린다. seconds 초 동안은 새로 들어오는 것도 버린다."""
        end = time.time() + max(0.0, seconds)
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                if time.time() >= end:
                    break
                time.sleep(0.02)

    def stop(self) -> None:
        self._stop.set()


class SpeechSegmenter:
    """소리 덩어리를 받아 '말이 시작됐다 / 끝났다'를 판단한다.

    마이크마다 조용할 때의 크기가 달라서, 주변 소음을 재어 두고 그보다 확실히
    큰 소리를 말소리로 본다. (새 마이크로 바꿔도 알아서 맞춘다)
    """

    def __init__(self, cfg: dict, logger: logging.Logger = log,
                 pre_roll_sec: float = 0.30, end_silence_sec: float | None = None,
                 max_sec: float | None = None, min_speech_sec: float = 0.25) -> None:
        self.cfg = cfg
        self.logger = logger
        self.block_sec = BLOCK_MS / 1000.0
        self.pre_roll = max(1, int(pre_roll_sec / self.block_sec))
        self.end_silence = max(2, int((end_silence_sec if end_silence_sec is not None
                                       else float(cfg.get("silence_sec") or 0.8)) / self.block_sec))
        self.max_blocks = max(4, int((max_sec if max_sec is not None
                                      else float(cfg.get("max_record_sec") or 12.0)) / self.block_sec))
        self.min_speech = max(1, int(min_speech_sec / self.block_sec))
        self.floor_threshold = float(cfg.get("speech_threshold") or 0.008)
        self.reset()

    def reset(self) -> None:
        self.noise = None
        self.speaking = False
        self.blocks: list[Any] = []
        self.pre: list[Any] = []
        self.silence_run = 0
        self.speech_run = 0
        self.speech_blocks = 0

    def _onset_threshold(self) -> float:
        noise = self.noise or 0.0
        return max(self.floor_threshold, noise * 2.6)

    def _end_threshold(self) -> float:
        noise = self.noise or 0.0
        return max(self.floor_threshold * 0.7, noise * 1.7)

    def feed(self, pcm) -> list[Any]:
        """블록 하나를 넣는다. 말이 끝났으면 그 말 전체(int16)를 목록으로 돌려준다."""
        np_mod = _need_numpy()
        level = _rms(pcm)
        done: list[Any] = []

        if self.noise is None:
            self.noise = level

        if not self.speaking:
            # 조용할 때만 주변 소음 추정치를 새로 한다
            if level < self._onset_threshold():
                self.noise = 0.97 * self.noise + 0.03 * level
            self.pre.append(pcm)
            if len(self.pre) > self.pre_roll:
                self.pre.pop(0)
            if level > self._onset_threshold():
                self.speech_run += 1
                if self.speech_run >= 2:
                    self.speaking = True
                    self.blocks = list(self.pre)
                    self.silence_run = 0
                    self.speech_blocks = len(self.blocks)
                    self.pre = []
            else:
                self.speech_run = 0
            return done

        self.blocks.append(pcm)
        if level > self._end_threshold():
            self.silence_run = 0
            self.speech_blocks += 1
        else:
            self.silence_run += 1

        finished = False
        if self.silence_run >= self.end_silence:
            finished = True
        elif len(self.blocks) >= self.max_blocks:
            finished = True

        if finished:
            if self.speech_blocks >= self.min_speech:
                audio = np_mod.concatenate(self.blocks) if self.blocks else np_mod.zeros(0, np_mod.int16)
                if audio.size:
                    done.append(audio)
            self.speaking = False
            self.blocks = []
            self.pre = []
            self.silence_run = 0
            self.speech_run = 0
            self.speech_blocks = 0
        return done

    def flush(self) -> list[Any]:
        """말하는 중이면 지금까지 모은 것을 내놓는다."""
        np_mod = _need_numpy()
        out = []
        if self.speaking and self.speech_blocks >= self.min_speech and self.blocks:
            out.append(np_mod.concatenate(self.blocks))
        self.speaking = False
        self.blocks = []
        self.pre = []
        self.silence_run = 0
        self.speech_run = 0
        self.speech_blocks = 0
        return out


# ---------------------------------------------------------------------------
# Vosk 1차 확인 (빠른 길)
# ---------------------------------------------------------------------------

def build_recognizer(model, cfg: dict, logger: logging.Logger = log):
    """깨우기 말만 알아듣는 인식기를 만든다."""
    vosk = _import_vosk(logger)
    grammar = wake_grammar(cfg)
    try:
        rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar)
    except Exception as exc:
        logger.warning("깨우기 문법을 쓰지 못해 전체 인식으로 넘어갑니다: %s", exc)
        rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    try:
        rec.SetWords(True)
    except Exception:
        pass
    return rec


@dataclass
class WakeHit:
    """깨우기 말을 들었을 때의 결과."""
    word: str
    text: str                       # 들은 말 전체 (받아쓰기)
    audio: Any = None               # 이번에 한 말 전체(int16)
    source: str = "vosk"            # vosk | whisper
    question: str = ""              # 부르는 말을 뺀 질문 (있으면)


def vosk_listen_segment(model, cfg: dict, pcm, logger: logging.Logger = log) -> tuple[str | None, str]:
    """말 한 토막을 Vosk 문법 인식기에 넣어 (깨우기 말, 들은 글자)를 돌려준다."""
    np_mod = _need_numpy()
    _import_vosk(logger)
    try:
        rec = build_recognizer(model, cfg, logger)
    except Exception as exc:
        logger.debug("Vosk 인식기를 만들지 못했습니다: %s", exc)
        return None, ""
    audio = np_mod.asarray(pcm, dtype=np_mod.int16).reshape(-1)
    texts: list[str] = []
    hit_text: str | None = None
    hit = None
    step = int(SAMPLE_RATE * BLOCK_MS / 1000)
    for start in range(0, max(1, audio.size), step):
        chunk = audio[start:start + step]
        if chunk.size == 0:
            continue
        try:
            if rec.AcceptWaveform(chunk.tobytes()):
                result = json.loads(rec.Result())
                text = str(result.get("text") or "").strip()
                if text:
                    texts.append(text)
                    if hit is None:
                        hit = find_wake(text, cfg, words=result.get("result"))
                        if hit:
                            hit_text = text
        except Exception as exc:
            logger.debug("Vosk 인식 오류: %s", exc)
            break
    # 끝을 알리는 무음 — 이것이 없으면 마지막 말이 결과에 안 들어온다
    try:
        tail = np_mod.zeros(int(SAMPLE_RATE * 0.4), dtype=np_mod.int16)
        rec.AcceptWaveform(tail.tobytes())
        result = json.loads(rec.FinalResult())
        text = str(result.get("text") or "").strip()
        if text:
            texts.append(text)
            if hit is None:
                hit = find_wake(text, cfg, words=result.get("result"))
                if hit:
                    hit_text = text
    except Exception:
        pass
    return hit, " ".join(texts).strip() or hit_text or ""


# ---------------------------------------------------------------------------
# 받아쓰기 (Whisper) + Vosk 대체
# ---------------------------------------------------------------------------

class WhisperNotReady(Exception):
    pass


class WhisperHolder:
    """받아쓰기 모델. 준비는 백그라운드로 하고, 준비되면 깨우기 2차 확인에도 쓴다."""

    def __init__(self) -> None:
        self.model = None
        self.ready = threading.Event()
        self.error: Exception | None = None
        self._lock = threading.Lock()
        self._loading = False
        self._last_try = 0.0
        self.load_seconds = 0.0
        self.transcribe_lock = threading.Lock()

    def start(self, cfg: dict, logger: logging.Logger = log) -> None:
        self._kick(cfg, logger)

    def _kick(self, cfg: dict, logger: logging.Logger = log) -> None:
        with self._lock:
            if self._loading:
                return
            self._loading = True
            self.ready.clear()
        self._last_try = time.time()
        threading.Thread(target=self._load, args=(cfg, logger), name="whisper-load", daemon=True).start()

    def _load(self, cfg: dict, logger: logging.Logger = log) -> None:
        try:
            from faster_whisper import WhisperModel
            name = str(cfg.get("whisper_model") or "small")
            logger.info("받아쓰기 모델(%s)을 준비합니다. 처음 한 번은 내려받느라 시간이 걸립니다.", name)
            started = time.time()
            model = WhisperModel(name, device="cpu", compute_type="int8",
                                 download_root=str(cache_dir() / "whisper"))
            with self._lock:
                self.model = model
                self.error = None
            self.load_seconds = time.time() - started
            self.ready.set()
            logger.info("받아쓰기 모델 준비 완료 (%.1f초)", self.load_seconds)
        except Exception as exc:
            with self._lock:
                self.error = exc
            self.ready.set()
            logger.warning("받아쓰기 모델 준비 실패(Vosk 로 대신합니다): %s", exc)
        finally:
            with self._lock:
                self._loading = False

    def get(self, cfg: dict, logger: logging.Logger = log, timeout: float = 0.0):
        if self.model is not None:
            return self.model
        if timeout > 0:
            self.ready.wait(timeout)
        if self.model is not None:
            return self.model
        if self.error is not None and not self._loading and (time.time() - self._last_try) > 180:
            self._kick(cfg, logger)
        return None

    def is_ready(self) -> bool:
        return self.model is not None


_WHISPER_NOISE = {
    "시청해주셔서 감사합니다.", "시청해 주셔서 감사합니다.", "구독과 좋아요 부탁드립니다.",
    "감사합니다.", "고맙습니다.", "음", "어", "네.", "네",
}


def normalize_audio(audio) -> Any:
    """받아쓰기 전에 소리 크기를 일정하게 맞춘다(너무 작은 소리도 잘 들리게)."""
    np_mod = _need_numpy()
    x = np_mod.asarray(audio, dtype=np_mod.float32).reshape(-1)
    if x.size == 0:
        return x
    x = x / 32768.0 if float(np_mod.max(np_mod.abs(x))) > 1.5 else x
    peak = float(np_mod.max(np_mod.abs(x)))
    if peak <= 1e-4:
        return x
    return x * (0.7 / peak)


def transcribe_whisper(audio, cfg: dict, holder: WhisperHolder,
                       logger: logging.Logger = log, timeout: float = 3.0) -> str:
    model = holder.get(cfg, logger, timeout=timeout)
    if model is None:
        raise WhisperNotReady("받아쓰기 모델이 아직 준비되지 않았습니다.")
    data = normalize_audio(audio)
    with holder.transcribe_lock:
        segments, _info = model.transcribe(
            data,
            language="ko",
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'").strip()
    if text in _WHISPER_NOISE:
        return ""
    return text


def transcribe_vosk(audio, holder: VoskHolder, cfg: dict, logger: logging.Logger = log) -> str:
    """Vosk 전체 모델로 받아쓴다(Whisper 를 못 쓸 때)."""
    np_mod = _need_numpy()
    vosk = _import_vosk(logger)
    model = holder.get(cfg, logger)
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    pcm = np_mod.asarray(audio, dtype=np_mod.int16).reshape(-1)
    step = int(SAMPLE_RATE * BLOCK_MS / 1000)
    texts: list[str] = []
    for start in range(0, max(1, pcm.size), step):
        chunk = pcm[start:start + step]
        if chunk.size == 0:
            continue
        try:
            if rec.AcceptWaveform(chunk.tobytes()):
                texts.append(str(json.loads(rec.Result()).get("text") or ""))
        except Exception:
            pass
    try:
        texts.append(str(json.loads(rec.FinalResult()).get("text") or ""))
    except Exception:
        pass
    text = re.sub(r"\s+", " ", " ".join(t for t in texts if t)).strip()
    return text if len(text) >= 2 else ""


# ---------------------------------------------------------------------------
# 목소리 (edge-tts → 윈도우 기본 음성)
# ---------------------------------------------------------------------------

def _edge_tts_save(text: str, voice: str, path: Path) -> None:
    import edge_tts

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(path))

    async def _with_timeout() -> None:
        await asyncio.wait_for(_run(), timeout=25)

    try:
        asyncio.run(_with_timeout())
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        raise


def _pygame_init() -> None:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()


def _play_mp3(path: Path, logger: logging.Logger = log) -> None:
    import pygame
    _pygame_init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    deadline = time.time() + 300
    while pygame.mixer.music.get_busy() and time.time() < deadline:
        time.sleep(0.05)
    try:
        pygame.mixer.music.unload()
    except Exception:
        pass


def speak_windows(text: str, logger: logging.Logger = log) -> None:
    """인터넷이나 edge-tts가 안 될 때 쓰는 윈도우 기본 음성."""
    text = clean_for_speech(text)
    if not text:
        return
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        logger.error("윈도우 기본 음성(powershell)을 찾지 못했습니다.")
        return
    txt_path = Path(tempfile.gettempdir()) / f"davis_tts_{os.getpid()}_{int(time.time()*1000)}.txt"
    ps_path = txt_path.with_suffix(".ps1")
    try:
        txt_path.write_text(text, encoding="utf-8")
        safe_path = str(txt_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech\r\n"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\r\n"
            "try { $s.SelectVoiceByHearingLocale('ko-KR') } catch {}\r\n"
            f"$t = Get-Content -Raw -Encoding UTF8 -LiteralPath '{safe_path}'\r\n"
            "if ($t) { $s.Speak($t) }\r\n"
            "$s.Dispose()\r\n"
        )
        ps_path.write_text(script, encoding="utf-8-sig")
        subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_path)],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=180,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception as exc:
        logger.warning("윈도우 기본 음성 재생 실패: %s", exc)
    finally:
        for p in (txt_path, ps_path):
            try:
                p.unlink()
            except Exception:
                pass


class Speaker:
    """목소리로 읽어 준다.

    - edge-tts(인터넷 음성) → 실패하면 윈도우 기본 음성
    - 문장 단위로 미리 만들어 두고 차례로 재생 (첫 문장부터 빨리 들리게)
    - 재생 중에는 '말하는 중' 상태를 알려 준다
    """

    def __init__(self, cfg: dict, logger: logging.Logger = log,
                 on_state: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.on_state = on_state
        self._edge_failed_until = 0.0
        self._lock = threading.Lock()
        self.speaking = False
        self.voice = str(cfg.get("tts_voice") or DEFAULT_CONFIG["tts_voice"])
        self.engine = str(cfg.get("tts_engine") or "auto")

    # -- 한 덩어리 만들기 ------------------------------------------------
    def _make_mp3(self, text: str, cache: bool = False) -> Path | None:
        """edge-tts 로 mp3 를 만든다. 못 만들면 None."""
        if self.engine == "windows":
            return None
        if time.time() < self._edge_failed_until:
            return None
        try:
            if cache:
                import hashlib
                digest = hashlib.sha1(f"{self.voice}|{text}".encode("utf-8")).hexdigest()[:16]
                path = cache_dir() / f"{digest}.mp3"
                if path.exists() and path.stat().st_size >= 400:
                    return path
            else:
                fd, name = tempfile.mkstemp(prefix="davis_ans_", suffix=".mp3")
                os.close(fd)
                path = Path(name)
            _edge_tts_save(text, self.voice, path)
            return path
        except Exception as exc:
            self.logger.warning("인터넷 음성(edge-tts) 실패: %s → 윈도우 기본 음성으로 대체합니다.", exc)
            self._edge_failed_until = time.time() + 120
            return None

    def _say_blocking(self, text: str, cache: bool = False) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        path = self._make_mp3(text, cache=cache)
        if path is None:
            speak_windows(text, self.logger)
            return
        try:
            _play_mp3(path, self.logger)
        except Exception as exc:
            self.logger.warning("인터넷 음성 재생 실패(%s) → 윈도우 기본 음성으로 대체", exc)
            speak_windows(text, self.logger)
        finally:
            if not cache:
                try:
                    path.unlink()
                except Exception:
                    pass

    # -- 공개 ------------------------------------------------------------
    def say(self, text: str, cache: bool = False) -> None:
        with self._lock:
            self.speaking = True
            self._state("말하는 중")
            try:
                self._say_blocking(text, cache=cache)
            finally:
                self.speaking = False
                self._state("듣는 중")

    def say_stream(self, chunks: Iterable[str], max_chars: int = 400) -> str:
        """답변 조각을 받는 대로 문장 단위로 읽어 준다. 읽은 전체 글자를 돌려준다."""
        max_chars = int(max_chars or 400)
        sentences: queue.Queue = queue.Queue()
        produced: list[str] = []
        errors: list[BaseException] = []
        DONE = object()

        def producer() -> None:
            buffer = ""
            spoken = 0
            try:
                for chunk in chunks:
                    if not chunk:
                        continue
                    produced.append(chunk)
                    buffer += chunk
                    if spoken + len(buffer) > max_chars:
                        buffer = buffer[: max(0, max_chars - spoken)]
                    ready, buffer = split_sentences(buffer)
                    for sentence in ready:
                        if spoken >= max_chars:
                            break
                        sentences.put(sentence)
                        spoken += len(sentence)
                    if spoken >= max_chars:
                        break
                rest = buffer.strip()
                if rest and spoken < max_chars:
                    sentences.put(rest)
            except BaseException as exc:
                self.logger.warning("답변을 받는 중 오류: %s", exc)
                errors.append(exc)
            finally:
                try:
                    if hasattr(chunks, "close"):
                        chunks.close()          # 클로드 프로세스를 바로 정리한다
                except Exception:
                    pass
                sentences.put(DONE)

        # 앞 문장의 소리를 만드는 동안 다음 문장 소리를 미리 만든다
        audio_queue: queue.Queue = queue.Queue(maxsize=3)
        AUDIO_DONE = object()

        def audio_maker() -> None:
            while True:
                item = sentences.get()
                if item is DONE:
                    audio_queue.put(AUDIO_DONE)
                    return
                text = clean_for_speech(str(item))
                if not text:
                    continue
                path = self._make_mp3(text)
                audio_queue.put((text, path))

        maker = threading.Thread(target=audio_maker, name="tts-make", daemon=True)
        prod = threading.Thread(target=producer, name="tts-text", daemon=True)
        with self._lock:
            self.speaking = True
            self._state("말하는 중")
            try:
                prod.start()
                maker.start()
                while True:
                    item = audio_queue.get()
                    if item is AUDIO_DONE:
                        break
                    text, path = item
                    if path is None:
                        speak_windows(text, self.logger)
                        continue
                    try:
                        _play_mp3(path, self.logger)
                    except Exception as exc:
                        self.logger.warning("재생 실패(%s) → 윈도우 기본 음성", exc)
                        speak_windows(text, self.logger)
                    finally:
                        try:
                            path.unlink()
                        except Exception:
                            pass
            finally:
                self.speaking = False
                self._state("듣는 중")
        if errors and not produced:
            raise errors[0]
        return "".join(produced)

    def _state(self, text: str) -> None:
        if self.on_state is not None:
            try:
                self.on_state(text)
            except Exception:
                pass


def beep(logger: logging.Logger = log) -> None:
    """짧은 '삐' 소리. 실패해도 조용히 넘어간다."""
    try:
        np_mod = _need_numpy()
        sd = _sounddevice()
        sr = 24000
        dur = 0.12
        n = int(sr * dur)
        t = np_mod.linspace(0.0, dur, n, endpoint=False)
        tone = 0.28 * np_mod.sin(2.0 * np_mod.pi * 880.0 * t)
        fade = max(1, int(sr * 0.01))
        ramp = np_mod.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
        sd.play(tone.astype(np_mod.float32), sr)
        sd.wait()
    except Exception as exc:
        logger.debug("비프음 재생 실패: %s", exc)


# ---------------------------------------------------------------------------
# 클로드 코드 호출
# ---------------------------------------------------------------------------

class ClaudeError(Exception):
    pass


class ClaudeNotFound(ClaudeError):
    pass


class ClaudeTimeout(ClaudeError):
    pass


class ClaudeAuthError(ClaudeError):
    pass


class ClaudeQuotaError(ClaudeError):
    pass


def resolve_claude(name: str) -> str | None:
    name = str(name or "claude").strip() or "claude"
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "claude.exe",
        home / ".local" / "bin" / "claude.cmd",
        home / ".local" / "bin" / "claude",
        home / "AppData" / "Roaming" / "npm" / "claude.cmd",
        home / "AppData" / "Roaming" / "npm" / "claude.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "nodejs" / "claude.cmd",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def build_prompt(question: str, cfg: dict) -> str:
    now = datetime.now()
    owner = str(cfg.get("owner_name") or "김용진")
    name = str(cfg.get("assistant_name") or "자비스")
    return (
        f"너는 '{name}'야. 대표님({owner} 님)의 음성 비서야. "
        "한국어로 3문장 이내, 말하듯이 짧게 답해. 목록·마크다운·이모지·코드블록 금지. "
        "모르면 모른다고 말해. "
        f"오늘은 {now.year}년 {now.month}월 {now.day}일 {WEEKDAYS[now.weekday()]}요일, "
        f"지금은 {now.hour:02d}시 {now.minute:02d}분이야. "
        f"질문: {question}"
    )


_AUTH_HINTS = ("login", "log in", "logged in", "auth", "api key", "credential", "oauth",
               "unauthorized", "not authenticated", "sign in")
_QUOTA_HINTS = ("credit balance", "credit", "quota", "rate limit", "429", "too many requests",
                "usage limit", "limit reached", "insufficient")


def classify_claude_error(text: str) -> ClaudeError:
    merged = (text or "").lower()
    if any(hint in merged for hint in _QUOTA_HINTS):
        return ClaudeQuotaError(text or "사용 한도(크레딧)가 부족합니다.")
    if any(hint in merged for hint in _AUTH_HINTS):
        return ClaudeAuthError(text or "로그인이 필요합니다.")
    return ClaudeError(text or "알 수 없는 오류")


class ClaudeClient:
    """`claude -p` 를 실행해 답을 받는다. 되는 판이면 스트리밍으로 받아 첫 문장부터 읽는다."""

    def __init__(self, cfg: dict, logger: logging.Logger = log) -> None:
        self.cfg = cfg
        self.logger = logger
        self.exe = resolve_claude(str(cfg.get("claude_exe") or "claude"))
        self._stream_supported: bool | None = None

    def available(self) -> bool:
        return bool(self.exe)

    def _cwd(self) -> str:
        cwd_text = str(self.cfg.get("claude_cwd") or "").strip()
        return str(_expand(cwd_text)) if cwd_text else str(Path.home())

    def _base_cmd(self, prompt: str) -> list[str]:
        cmd = [self.exe or "claude", "-p", prompt]
        model = str(self.cfg.get("claude_model") or "").strip()
        if model:
            cmd += ["--model", model]
        return cmd

    def version(self) -> str:
        if not self.exe:
            return ""
        try:
            result = subprocess.run([self.exe, "--version"], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=60,
                                    stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            text = (result.stdout or result.stderr or "").strip()
            return text.splitlines()[0] if text else ""
        except Exception:
            return ""

    def ask_stream(self, question: str) -> Iterator[str]:
        """답변 글자를 조금씩 내놓는다. 실패하면 ClaudeError."""
        if not self.exe:
            raise ClaudeNotFound("claude 실행 파일을 찾지 못했습니다.")
        prompt = build_prompt(question, self.cfg)
        try:
            timeout = float(self.cfg.get("claude_timeout_sec") or 180)
        except (TypeError, ValueError):
            timeout = 180.0

        if self._stream_supported is False:
            yield from self._plain(prompt, timeout)
            return

        got = False
        try:
            for text in self._stream(prompt, timeout):
                got = True
                yield text
            self._stream_supported = True
        except _StreamUnsupported:
            self.logger.info("이 클로드 코드 판은 스트리밍 출력을 지원하지 않아 통째로 받습니다.")
            self._stream_supported = False
            yield from self._plain(prompt, timeout)
        except ClaudeError:
            raise
        if got:
            return

    def _stream(self, prompt: str, timeout: float) -> Iterator[str]:
        cmd = self._base_cmd(prompt) + [
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        ]
        process = None
        try:
            process = subprocess.Popen(
                cmd, cwd=self._cwd(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW, bufsize=1,
            )
        except OSError as exc:
            raise ClaudeNotFound(str(exc))

        deadline = time.time() + timeout
        produced = 0
        stderr_lines: list[str] = []

        def read_err() -> None:
            try:
                for line in process.stderr:     # type: ignore[union-attr]
                    stderr_lines.append(line.rstrip())
                    if len(stderr_lines) > 40:
                        stderr_lines.pop(0)
            except Exception:
                pass

        err_thread = threading.Thread(target=read_err, name="claude-err", daemon=True)
        err_thread.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if time.time() > deadline:
                    raise ClaudeTimeout(f"{timeout}초 안에 답이 끝나지 않았습니다.")
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    # stream-json 이 아닌 평범한 글자가 오면 그대로 내놓는다
                    if not line.startswith("{"):
                        produced += len(line)
                        yield line + " "
                    continue
                kind = event.get("type")
                if kind == "stream_event":
                    inner = event.get("event") or {}
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            produced += len(delta["text"])
                            yield str(delta["text"])
                elif kind == "assistant":
                    message = event.get("message") or {}
                    for block in message.get("content") or []:
                        if block.get("type") == "text" and block.get("text"):
                            if produced == 0:
                                produced += len(block["text"])
                                yield str(block["text"])
                elif kind == "result":
                    if produced == 0 and event.get("result"):
                        produced += len(str(event["result"]))
                        yield str(event["result"])
                    if event.get("is_error"):
                        raise classify_claude_error(str(event.get("result") or ""))
            try:
                code = process.wait(timeout=max(5.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                raise ClaudeTimeout(f"{timeout}초 안에 답이 끝나지 않았습니다.")
            merged = " ".join(stderr_lines)
            if code not in (0, None) and produced == 0:
                if "unknown option" in merged.lower() or "unrecognized" in merged.lower() \
                        or "invalid option" in merged.lower():
                    raise _StreamUnsupported()
                raise classify_claude_error(merged)
            if produced == 0 and not merged:
                raise ClaudeError(f"클로드가 아무 답도 내놓지 않았습니다(종료 코드 {code}).")
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass

    def _plain(self, prompt: str, timeout: float) -> Iterator[str]:
        cmd = self._base_cmd(prompt) + ["--output-format", "text"]
        try:
            result = subprocess.run(
                cmd, cwd=self._cwd(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout, creationflags=CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired:
            raise ClaudeTimeout(f"{timeout}초 안에 답이 오지 않았습니다.")
        except OSError as exc:
            raise ClaudeNotFound(str(exc))
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0 or not out:
            raise classify_claude_error(err or out or f"claude 종료 코드 {result.returncode}")
        yield out


class _StreamUnsupported(Exception):
    pass


# ---------------------------------------------------------------------------
# 자가진단
# ---------------------------------------------------------------------------

@dataclass
class CheckItem:
    level: str          # ok | warn | bad
    title: str
    detail: str = ""
    key: str = ""

    def line(self) -> str:
        mark = {"ok": "[정상]", "warn": "[주의]", "bad": "[문제]"}.get(self.level, "[정보]")
        return f"  {mark} {self.title}" + (f" — {self.detail}" if self.detail else "")


def _dep_status(module_name: str, label: str) -> tuple[bool, str]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "")
        return True, f"{label} {version}".strip()
    except Exception as exc:
        return False, f"{label} 없음/오류 ({exc.__class__.__name__})"


def mic_level_probe(cfg: dict, logger: logging.Logger = log, seconds: float = 2.5) -> tuple[bool, str]:
    """마이크를 잠깐 열어 소리 크기를 재어 본다."""
    try:
        stream = MicStream(cfg, logger).open()
    except MicError as exc:
        return False, str(exc)
    _need_numpy()
    levels: list[float] = []
    try:
        started = time.time()
        while time.time() - started < seconds:
            levels.append(_rms(stream.read_block()))
    except Exception as exc:
        stream.close()
        return False, f"마이크에서 소리를 읽지 못했습니다: {exc}"
    finally:
        stream.close()
    if not levels:
        return False, "마이크에서 소리를 받지 못했습니다."
    peak = max(levels)
    quiet = sorted(levels)[len(levels) // 2]
    detail = f"{stream.device_name} · 조용할 때 {quiet:.3f} / 가장 큰 소리 {peak:.3f}"
    if peak < 0.0015:
        return False, detail + " — 소리가 거의 0입니다. 마이크가 음소거됐거나 다른 장치를 쓰고 있습니다."
    if peak < 0.01:
        return True, detail + " — 소리가 작습니다. 마이크를 입 가까이 두거나 mic_gain 을 2.0으로 올려 보세요."
    return True, detail


def run_selfcheck(cfg: dict, logger: logging.Logger = log, deep: bool = False) -> list[CheckItem]:
    """자가진단. deep=True 이면 모델을 실제로 읽어 보고 마이크도 길게 잰다."""
    """시작할 때 보는 자가진단: 음성모델 / 마이크 / 클로드 코드 (+ 패키지·설정)."""
    items: list[CheckItem] = []

    # 0) 설정 파일
    if CONFIG_PATH.exists():
        items.append(CheckItem("ok", "설정 파일", str(CONFIG_PATH), "config"))
        try:
            json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            items.append(CheckItem("bad", "config.json 형식이 깨졌습니다",
                                   f"{exc} — install.bat 을 다시 실행하면 새로 만들어집니다", "config"))
    else:
        items.append(CheckItem("warn", "config.json 이 없습니다",
                               "install.bat 또는 --setup 을 실행해 주세요", "config"))

    # 1) 음성모델
    model_dir = find_model_dir(cfg)
    if model_dir is None:
        items.append(CheckItem("bad", "한국어 음성모델이 없습니다",
                               "install.bat 을 다시 실행하거나 --setup 을 실행해 주세요 (약 87MB)", "model"))
    else:
        items.append(CheckItem("ok", "한국어 음성모델", f"{model_dir} ({model_size_mb(model_dir):.0f}MB)", "model"))
        if deep:
            try:
                vosk_mod = _import_vosk(logger)
                started = time.time()
                loaded = vosk_mod.Model(str(model_dir))
                items.append(CheckItem("ok", "모델 읽기", f"{time.time() - started:.1f}초", "model"))
                del loaded
            except Exception as exc:
                items.append(CheckItem("bad", "모델을 읽지 못했습니다", str(exc), "model"))

    # 2) 마이크
    try:
        ok, detail = mic_level_probe(cfg, logger, seconds=2.0 if not deep else 3.0)
        items.append(CheckItem("ok" if ok else "bad", "마이크", detail, "mic"))
    except Exception as exc:
        items.append(CheckItem("bad", "마이크를 확인하지 못했습니다", str(exc), "mic"))

    # 3) 클로드 코드 (답을 만드는 쪽)
    exe = resolve_claude(str(cfg.get("claude_exe") or "claude"))
    if not exe:
        items.append(CheckItem("bad", "클로드 코드를 찾지 못했습니다",
                               "nodejs.org 에서 Node.js(LTS)를 설치한 뒤 'npm install -g @anthropic-ai/claude-code'", "claude"))
    else:
        version = ClaudeClient(cfg, logger).version()
        items.append(CheckItem("ok", "클로드 코드", f"{exe}" + (f" ({version})" if version else ""), "claude"))
        items.append(CheckItem("warn", "클로드 코드 로그인",
                               "첫 질문을 해 봐야 알 수 있습니다. 안 되면 cmd 에서 'claude' 실행 후 /login", "claude"))

    # 4) 패키지
    missing = []
    for module_name, label in (("numpy", "numpy"), ("sounddevice", "sounddevice"), ("vosk", "vosk"),
                               ("faster_whisper", "faster-whisper"), ("edge_tts", "edge-tts"),
                               ("pygame", "pygame")):
        good, text = _dep_status(module_name, label)
        if not good:
            missing.append(text)
    if missing:
        items.append(CheckItem("bad", "필요한 프로그램이 빠졌습니다", ", ".join(missing) + " — install.bat 을 다시 실행", "deps"))
    else:
        items.append(CheckItem("ok", "필요한 프로그램", "모두 있음", "deps"))

    # 5) 로그 폴더
    folder = log_dir()
    items.append(CheckItem("ok" if _can_write(folder) else "warn", "로그 폴더", str(folder), "log"))
    return items


# ---------------------------------------------------------------------------
# 대화
# ---------------------------------------------------------------------------

class Assistant:
    """마이크 → 깨우기 → 질문 → 클로드 → 목소리 를 잇는 본체."""

    def __init__(self, cfg: dict, logger: logging.Logger = log,
                 on_state: Callable[[str], None] | None = None,
                 on_heard: Callable[[str], None] | None = None,
                 on_answer: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.on_state = on_state
        self.on_heard = on_heard
        self.on_answer = on_answer
        self.stop_event = threading.Event()
        self.whisper = WhisperHolder()
        self.vosk = VoskHolder()
        self.speaker = Speaker(cfg, logger, on_state=on_state)
        self.reader: MicReader | None = None
        self.stream: MicStream | None = None
        self.last_interaction = 0.0
        self.critical_notice_done = False

    # -- 상태 알림 --------------------------------------------------------
    def _state(self, text: str) -> None:
        if self.on_state is not None:
            try:
                self.on_state(text)
            except Exception:
                pass

    def _heard(self, text: str) -> None:
        if self.on_heard is not None:
            try:
                self.on_heard(text)
            except Exception:
                pass

    def _answer(self, text: str) -> None:
        if self.on_answer is not None:
            try:
                self.on_answer(text)
            except Exception:
                pass

    # -- 시작 -------------------------------------------------------------
    def start_mic(self) -> None:
        self.stream = MicStream(self.cfg, self.logger).open()
        self.reader = MicReader(self.stream, self.logger)
        self.reader.start()

    def _ensure_mic(self) -> bool:
        """마이크가 죽었으면 다시 연다. 성공하면 True."""
        if self.reader is not None and self.reader.error is None:
            return True
        if self.stream is not None:
            self.stream.close()
        for attempt in range(3):
            if self.stop_event.is_set():
                return False
            try:
                self.start_mic()
                self.logger.info("마이크를 다시 열었습니다.")
                return True
            except Exception as exc:
                self.logger.warning("마이크 다시 열기 실패(%d/3): %s", attempt + 1, exc)
                time.sleep(2)
        return False

    def engine(self) -> str:
        value = str(self.cfg.get("wake_engine") or "hybrid").lower()
        return value if value in ("hybrid", "fast", "accurate") else "hybrid"

    # -- 깨우기 -----------------------------------------------------------
    def wait_for_wake(self) -> WakeHit | None:
        """깨우기 말이 들릴 때까지 마이크를 듣는다. 멈추면 None."""
        assert self.reader is not None
        segmenter = SpeechSegmenter(self.cfg, self.logger, end_silence_sec=0.55, max_sec=8.0,
                                    min_speech_sec=0.25)
        model = None
        engine = self.engine()
        if engine != "accurate":
            try:
                model = self.vosk.get(self.cfg, self.logger)
            except Exception as exc:
                self.logger.warning("Vosk 준비 실패(Whisper 로만 듣습니다): %s", exc)
                model = None
                if engine == "fast":
                    self._speak_problem("부르는 말을 알아듣는 준비가 안 됐어요. 설치 파일을 다시 실행해 주세요.")
                    engine = "accurate"

        last_status = 0.0
        while not self.stop_event.is_set():
            pcm = self.reader.get(timeout=0.5)
            if self.reader.error is not None:
                raise MicError(str(self.reader.error))
            if pcm is None:
                if time.time() - last_status > 30:
                    last_status = time.time()
                    self._state("듣는 중")
                continue
            if time.time() < self.last_interaction + float(self.cfg.get("wake_cooldown_sec") or 3.0):
                continue
            for segment in segmenter.feed(pcm):
                hit = self._check_segment(segment, model, engine)
                if hit is not None:
                    return hit
        return None

    def _check_segment(self, segment, model, engine: str) -> WakeHit | None:
        """말 한 토막이 깨우기 말인지 본다. (Vosk 1차 → Whisper 2차)"""
        _need_numpy()
        speech_sec = segment.size / float(SAMPLE_RATE)
        if speech_sec < 0.25:
            return None
        vosk_text = ""
        vosk_hit = None
        if model is not None and engine in ("fast", "hybrid"):
            try:
                vosk_hit, vosk_text = vosk_listen_segment(model, self.cfg, segment, self.logger)
            except Exception as exc:
                self.logger.debug("Vosk 1차 확인 실패: %s", exc)
        if vosk_hit:
            self.logger.info("깨우기(Vosk): '%s' — 들은 말: %s", vosk_hit, vosk_text)
            self._heard(f"[Vosk] {vosk_text or vosk_hit}")
            return WakeHit(word=vosk_hit, text=vosk_text or vosk_hit, audio=segment, source="vosk")

        if engine == "fast":
            if vosk_text:
                self.logger.debug("Vosk 가 들었지만 깨우기 말은 아닙니다: %s", vosk_text)
            return None

        # 2차: Whisper. (시끄러운 곳에서 Vosk 가 놓친 것을 건진다)
        text = ""
        try:
            if self.whisper.is_ready():
                text = transcribe_whisper(segment, self.cfg, self.whisper, self.logger, timeout=5.0)
            elif self.whisper.error is not None:
                text = transcribe_vosk(segment, self.vosk, self.cfg, self.logger) if model is not None else ""
        except Exception as exc:
            self.logger.debug("Whisper 2차 확인 실패: %s", exc)
            text = vosk_text
        text = text or vosk_text
        hit = whisper_wake_hit(text, self.cfg) if text else None
        if hit:
            self.logger.info("깨우기(Whisper): '%s' — 들은 말: %s", hit, text)
            self._heard(text)
            question = strip_wake_prefix(text, self.cfg)
            return WakeHit(word=hit, text=text, audio=segment, source="whisper", question=question)
        if text:
            # 무응답의 원인을 알 수 있게, 들은 말은 남긴다(질문 내용은 남기지 않는다)
            self.logger.debug("말은 들었지만 깨우기 말이 아닙니다: %s", text)
            self._heard(f"(깨우기 아님) {text[:40]}")
        return None

    # -- 질문 듣기 --------------------------------------------------------
    def listen_question(self, wait_sec: float | None = None) -> tuple[Any, str]:
        """부른 뒤의 질문을 듣는다. (소리, 받아쓴 글자)"""
        assert self.reader is not None
        wait_sec = float(wait_sec if wait_sec is not None else (self.cfg.get("listen_wait_sec") or 6.0))
        segmenter = SpeechSegmenter(self.cfg, self.logger, min_speech_sec=0.2)
        self.reader.drain()
        started = time.time()
        self._state("듣는 중 (질문)")
        while not self.stop_event.is_set() and time.time() - started < wait_sec:
            pcm = self.reader.get(timeout=0.3)
            if pcm is None:
                continue
            for segment in segmenter.feed(pcm):
                return segment, ""
        segments = segmenter.flush()
        if segments:
            return segments[0], ""
        return None, ""

    # -- 한 번의 대화 -----------------------------------------------------
    def interact(self, hit: WakeHit) -> None:
        self.last_interaction = time.time()
        beep(self.logger)
        self.speaker.say(str(self.cfg.get("wake_reply") or "네, 대표님"), cache=True)

        question = (hit.question or "").strip()
        if len(question) < 2:
            if self.reader is not None:
                self.reader.drain()
            audio, _text = self.listen_question()
            if audio is None:
                self.speaker.say("듣지 못했어요. 다시 불러 주세요.", cache=True)
                return
            question = self._transcribe(audio)
        if not question:
            self.speaker.say("잘 못 들었어요. 다시 말씀해 주세요.", cache=True)
            return
        self._heard(question)
        self.logger.info("질문 인식 완료 (%d자)", len(question))

        client = ClaudeClient(self.cfg, self.logger)
        if not client.available():
            self.logger.error("claude 실행 파일을 찾지 못했습니다.")
            self.speaker.say("클로드 코드가 설치되지 않았어요. 설치 안내를 확인해 주세요.", cache=True)
            return

        self._state("답 만드는 중")
        started = time.time()
        try:
            if bool(self.cfg.get("stream_answer", True)):
                answer = self.speaker.say_stream(client.ask_stream(question),
                                                 int(self.cfg.get("max_answer_chars") or 400))
            else:
                text = "".join(client.ask_stream(question))
                answer = trim_answer(text, int(self.cfg.get("max_answer_chars") or 400))
                self.speaker.say(answer)
        except ClaudeNotFound:
            self.logger.error("claude 실행 파일을 찾지 못했습니다.")
            self.speaker.say("클로드 코드가 설치되지 않았어요. 설치 안내를 확인해 주세요.", cache=True)
            return
        except ClaudeAuthError as exc:
            self.logger.error("claude 로그인 문제: %s", exc)
            self.speaker.say("클로드 코드 로그인이 풀린 것 같아요. 컴퓨터에서 클로드를 실행해 다시 로그인해 주세요.", cache=True)
            return
        except ClaudeQuotaError as exc:
            self.logger.error("claude 사용 한도 문제: %s", exc)
            self.speaker.say("클로드 사용 한도가 다 된 것 같아요. 결제나 사용량을 확인해 주세요.", cache=True)
            return
        except ClaudeTimeout as exc:
            self.logger.warning("claude 응답 시간 초과: %s", exc)
            self.speaker.say("답을 만드는 데 시간이 너무 오래 걸렸어요. 다시 물어봐 주세요.", cache=True)
            return
        except ClaudeError as exc:
            self.logger.error("claude 오류: %s", exc)
            self.speaker.say("답을 만들지 못했어요. 잠시 뒤에 다시 물어봐 주세요.", cache=True)
            return

        if not (answer or "").strip():
            self.logger.warning("클로드가 빈 답을 내놓았습니다.")
            self.speaker.say("답을 받지 못했어요. 다시 물어봐 주세요.", cache=True)
            return

        log_text = trim_answer(answer, 200)
        self.logger.info("답변 전달 완료 (%d자, %.1f초)", len(answer), time.time() - started)
        self._answer(log_text)
        self.last_interaction = time.time()

    def _transcribe(self, audio) -> str:
        """받아쓴다. Whisper 가 아직 준비 안 됐으면 Vosk 로 받아쓴다."""
        whisper_error: Exception | None = None
        try:
            text = transcribe_whisper(audio, self.cfg, self.whisper, self.logger, timeout=8.0)
            if text:
                return text
        except Exception as exc:
            whisper_error = exc
            self.logger.warning("Whisper 받아쓰기를 쓰지 못했습니다: %s", exc)
        try:
            text = transcribe_vosk(audio, self.vosk, self.cfg, self.logger)
            if text:
                self.logger.info("Vosk 로 받아썼습니다: %d자", len(text))
                return text
        except Exception as exc:
            self.logger.warning("Vosk 받아쓰기 실패: %s", exc)
        if isinstance(whisper_error, WhisperNotReady):
            self.speaker.say("받아쓰기 준비 중이에요. 처음 한 번은 파일을 받느라 몇 분 걸립니다. 잠시 뒤 다시 불러 주세요.", cache=True)
        return ""

    # -- 본 루프 ----------------------------------------------------------
    def run(self) -> int:
        """멈출 때까지 듣고 대답한다. 반환값은 종료 코드."""
        self.logger.info("자비스 시작 (v%s, 엔진=%s)", APP_VERSION, self.engine())
        self.whisper.start(self.cfg, self.logger)

        # 자가진단 — 문제가 있으면 한 번은 소리로도 알려 준다
        items = run_selfcheck(self.cfg, self.logger)
        for item in items:
            self.logger.log(logging.ERROR if item.level == "bad" else logging.INFO, "%s", item.line().strip())
        bad = [i for i in items if i.level == "bad"]
        if bad and not self.critical_notice_done:
            self.critical_notice_done = True
            self._notify_problem(bad)

        if bool(self.cfg.get("announce_start")):
            self.speaker.say("자비스 준비됐어요.", cache=True)

        backoff = 1.0
        while not self.stop_event.is_set():
            if not self._ensure_mic():
                self._state("마이크 없음")
                if self.stop_event.wait(10):
                    break
                continue
            try:
                self._state("듣는 중")
                hit = self.wait_for_wake()
                if hit is None:
                    continue
                backoff = 1.0
                self.interact(hit)
                if self.reader is not None:
                    self.reader.drain(seconds=0.3)
            except MicError as exc:
                self.logger.error("마이크 문제: %s", exc)
                self._state("마이크 오류")
                if self.reader is not None:
                    self.reader.error = exc
                self._notify_mic_problem(str(exc))
                if self.stop_event.wait(min(30.0, backoff)):
                    break
                backoff = min(30.0, backoff * 2)
            except ModelError as exc:
                self.logger.error("음성모델 문제: %s", exc)
                self._notify_problem([CheckItem("bad", "음성모델", str(exc))])
                if self.stop_event.wait(60):
                    break
            except Exception as exc:
                self.logger.exception("대화 처리 중 오류: %s", exc)
                self._state("오류 — 다시 시작합니다")
                if self.stop_event.wait(2):
                    break
        if self.stream is not None:
            self.stream.close()
        self.logger.info("자비스 종료")
        return EXIT_USER_QUIT if self.stop_event.is_set() else 0

    def _speak_problem(self, text: str) -> None:
        self.logger.warning("안내: %s", text)
        try:
            self.speaker.say(text, cache=True)
        except Exception:
            pass

    def _notify_problem(self, bad: list[CheckItem]) -> None:
        """치명적인 문제는 한 번만 소리로 알린다(같은 말 반복 방지)."""
        parts = [f"{item.title}" for item in bad[:2]]
        message = "자비스 준비에 문제가 있어요. " + ", ".join(parts) + "."
        self._speak_problem(message)

    def _notify_mic_problem(self, detail: str) -> None:
        now = time.time()
        if getattr(self, "_last_mic_notice", 0) and now - getattr(self, "_last_mic_notice", 0) < 120:
            return
        self._last_mic_notice = now
        self._speak_problem("마이크를 찾지 못했어요. 마이크가 꽂혀 있는지 확인해 주세요.")

    def stop(self) -> None:
        self.stop_event.set()
        if self.reader is not None:
            self.reader.stop()
        if self.stream is not None:
            self.stream.close()


# ---------------------------------------------------------------------------
# 상태 창 (tkinter)
# ---------------------------------------------------------------------------

class StatusWindow:
    """자비스가 살아 있는지, 무엇을 들었는지 눈으로 볼 수 있는 작은 창.

    창을 닫으면 작업 표시줄로 내려가고, 자비스는 계속 듣습니다.
    (완전히 끄려면 [종료] 버튼)
    """

    def __init__(self, assistant: Assistant, cfg: dict, logger: logging.Logger = log,
                 on_quit: Callable[[], None] | None = None) -> None:
        self.assistant = assistant
        self.cfg = cfg
        self.logger = logger
        self.on_quit = on_quit
        self.events: queue.Queue = queue.Queue()
        self.should_close = False
        self.root = None
        self.status_var = None
        self.heard_var = None
        self.answer_var = None
        self.mic_var = None

    def update_state(self, text: str) -> None:
        self.events.put(("state", text))

    def update_heard(self, text: str) -> None:
        self.events.put(("heard", text))

    def update_answer(self, text: str) -> None:
        self.events.put(("answer", text))

    def _build(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        self.root = root
        root.title(f"{APP_TITLE} — 듣는 중")
        root.geometry("460x300")
        root.minsize(380, 240)
        try:
            root.attributes("-topmost", False)
        except Exception:
            pass

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)

        title = ttk.Label(frame, text="자비스", font=("Malgun Gothic", 16, "bold"))
        title.pack(anchor="w")
        self.status_var = tk.StringVar(value="시작하는 중...")
        ttk.Label(frame, textvariable=self.status_var, font=("Malgun Gothic", 12)).pack(anchor="w", pady=(4, 8))

        self.mic_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.mic_var, foreground="#555").pack(anchor="w")

        box = ttk.LabelFrame(frame, text="마지막으로 들은 말", padding=8)
        box.pack(fill="x", pady=(10, 6))
        self.heard_var = tk.StringVar(value="(아직 없음)")
        ttk.Label(box, textvariable=self.heard_var, wraplength=400, foreground="#0a4").pack(anchor="w")

        box2 = ttk.LabelFrame(frame, text="마지막 답변", padding=8)
        box2.pack(fill="both", expand=True, pady=(0, 6))
        self.answer_var = tk.StringVar(value="(아직 없음)")
        ttk.Label(box2, textvariable=self.answer_var, wraplength=400, justify="left").pack(anchor="w")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="소리 테스트", command=self._test_voice).pack(side="left")
        ttk.Button(buttons, text="로그 열기", command=self._open_log).pack(side="left", padx=4)
        ttk.Button(buttons, text="폴더 열기", command=self._open_folder).pack(side="left")
        ttk.Button(buttons, text="종료", command=self._quit).pack(side="right")

        root.protocol("WM_DELETE_WINDOW", self._hide)
        root.after(120, self._pump)

    def _pump(self) -> None:
        if self.should_close:
            self._destroy()
            return
        try:
            while True:
                kind, text = self.events.get_nowait()
                if kind == "state":
                    self.status_var.set(text if text else "듣는 중")
                    if self.root is not None:
                        self.root.title(f"{APP_TITLE} — {text}")
                elif kind == "heard":
                    self.heard_var.set(text)
                elif kind == "answer":
                    self.answer_var.set(text)
        except queue.Empty:
            pass
        if self.root is not None:
            self.root.after(150, self._pump)

    def _test_voice(self) -> None:
        threading.Thread(target=lambda: self.assistant.speaker.say("자비스 목소리 시험입니다.", cache=True),
                         daemon=True).start()

    def _open_log(self) -> None:
        try:
            folder = log_dir()
            files = sorted(folder.glob("????-??-??.log"))
            target = files[-1] if files else folder
            os.startfile(str(target))          # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.warning("로그를 열지 못했습니다: %s", exc)

    def _open_folder(self) -> None:
        try:
            os.startfile(str(BASE_DIR))         # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.warning("폴더를 열지 못했습니다: %s", exc)

    def _hide(self) -> None:
        try:
            assert self.root is not None
            self.root.iconify()
            self.logger.info("상태 창을 작업 표시줄로 내렸습니다(자비스는 계속 듣습니다).")
        except Exception:
            pass

    def _quit(self) -> None:
        if self.on_quit is not None:
            try:
                self.on_quit()
            except Exception:
                pass
        self._destroy()

    def _destroy(self) -> None:
        try:
            if self.root is not None:
                self.root.destroy()
                self.root = None
        except Exception:
            pass

    def run(self) -> None:
        """이미 만든 창의 이벤트 루프를 돈다. (_build 를 먼저 부른다)"""
        if self.root is not None:
            self.root.mainloop()


# ---------------------------------------------------------------------------
# 명령들
# ---------------------------------------------------------------------------

def _progress_printer():
    last = [0]

    def show(done: int, total: int) -> None:
        mb = done / 1024 / 1024
        if total:
            pct = int(done * 100 / total)
            if pct != last[0]:
                last[0] = pct
                _p(f"\r   내려받는 중... {pct:3d}%  ({mb:5.1f} / {total / 1024 / 1024:.0f}MB)", end="")
        else:
            _p(f"\r   내려받는 중... {mb:5.1f}MB", end="")

    return show


def cmd_setup(cfg: dict, model_path: str | None = None) -> int:
    _p("")
    _p("=" * 60)
    _p("  자비스 설정 만들기 (가입·키 필요 없음)")
    _p("=" * 60)
    _p("")

    if not CONFIG_PATH.exists():
        try:
            save_config(cfg)
            _p(f"   [완료] 설정 파일을 만들었습니다: {CONFIG_PATH}")
        except Exception as exc:
            _p(f"   [문제] 설정 파일을 만들지 못했습니다: {exc}")
            _p("          폴더를 C:\\davis 처럼 단순한 곳에 두고 다시 실행해 보세요.")
            return 1
    else:
        _p(f"   [확인] 설정 파일: {CONFIG_PATH}")

    if model_path:
        candidate = _expand(model_path)
        if not is_model_dir(candidate):
            _p(f"   [문제] 그 폴더에서 음성모델을 찾지 못했습니다: {candidate}")
            _p("          폴더 안에 am\\final.mdl 과 conf\\model.conf 가 있어야 합니다.")
            return 1
        cfg["vosk_model_path"] = str(candidate)
        save_config(cfg)
        _p(f"   [완료] 음성모델 위치를 저장했습니다: {candidate}")
        _print_setup_done(cfg)
        return 0

    found = find_model_dir(cfg)
    if found is not None:
        if str(cfg.get("vosk_model_path") or "").strip() != str(found):
            cfg["vosk_model_path"] = str(found)
            save_config(cfg)
        _p(f"   [확인] 한국어 음성모델: {found}")
        _print_setup_done(cfg)
        return 0

    _p(f"   한국어 음성모델을 내려받습니다. 약 {VOSK_MODEL_MB}MB, 1~3분 걸립니다.")
    _p("   (한 번만 받으면 됩니다. 창이 멈춘 것처럼 보여도 기다려 주세요.)")
    try:
        dest = ensure_model(cfg, log, on_progress=_progress_printer())
    except Exception as exc:
        _p("")
        _p(f"   [문제] 음성모델을 자동으로 받지 못했습니다: {exc}")
        _p("")
        _p("   인터넷이 되는 브라우저에서 아래 주소를 열어 파일을 받으세요.")
        _p(f"   {VOSK_MODEL_URL}")
        _p("")
        _p("   받은 압축 파일을 풀면 'vosk-model-small-ko-0.22' 폴더가 나옵니다.")
        _p("   그 폴더를 이 프로그램 폴더에 넣고 다시 실행하면 됩니다.")
        _p('   .venv\\Scripts\\python davis.py --setup --model-path "D:\\vosk-model-small-ko-0.22"')
        _p("")
        return 1

    _p("")
    _p(f"   [완료] 음성모델을 받았습니다: {dest}")
    _print_setup_done(cfg)
    return 0


def _print_setup_done(cfg: dict) -> None:
    words = ", ".join(f"'{w}'" for w in (cfg.get("wake_words") or []))
    _p("")
    _p("   부르는 말: " + (words or "'자비스'"))
    _p("   다음 단계: install.bat 을 마저 실행하거나, '자비스 시작.vbs' 를 더블클릭하세요.")
    _p("   상태를 확인하려면: .venv\\Scripts\\python davis.py --check")
    _p("")


def cmd_check(cfg: dict) -> int:
    _p("")
    _p("=" * 62)
    _p(f"  자비스 상태 점검 (v{APP_VERSION})")
    _p("=" * 62)
    items = run_selfcheck(cfg, log, deep=True)
    group_titles = {
        "config": "\n[1] 설정 파일",
        "model": "\n[2] 한국어 음성모델 (부르는 말을 알아듣는 데 씁니다)",
        "mic": "\n[3] 마이크",
        "claude": "\n[4] 클로드 코드 (답을 만드는 데 씁니다)",
        "deps": "\n[5] 필요한 프로그램",
        "log": "\n[6] 로그",
    }
    shown: set[str] = set()
    for item in items:
        if item.key not in shown:
            _p(group_titles.get(item.key, f"\n[{item.key}]"))
            shown.add(item.key)
        _p(item.line())

    _p("\n[7] 마이크 목록")
    for line in devices_report(cfg).splitlines():
        _p("  " + line)

    _p("\n[8] 음성 출력")
    try:
        speaker = Speaker(cfg, log)
        path = speaker._make_mp3("자비스 점검 중입니다.")
        if path is not None:
            _p("  [정상] 인터넷 음성(edge-tts) 사용 가능")
            try:
                path.unlink()
            except Exception:
                pass
        else:
            _p("  [주의] 인터넷 음성(edge-tts) 실패 → 윈도우 기본 음성으로 읽습니다.")
    except Exception as exc:
        _p(f"  [주의] 음성 출력 확인 실패: {exc}")

    _p("\n[9] 컴퓨터 켤 때 자동 시작")
    try:
        startup = startup_dir()
        if startup is None:
            _p("  [주의] 시작프로그램 폴더를 찾지 못했습니다(윈도우가 아닌 것 같습니다).")
        else:
            shortcut = startup / STARTUP_SHORTCUT_NAME
            if shortcut.exists():
                _p(f"  [정상] 자동 시작 바로가기 있음 ({shortcut})")
            else:
                _p("  [주의] 자동 시작 바로가기가 없습니다. install.bat 을 다시 실행하면 만들어집니다.")
            for item in stale_startup_items(startup):
                _p(f"  [주의] {item}")
    except Exception as exc:
        _p(f"  [주의] 자동 시작 항목을 확인하지 못했습니다: {exc}")

    _p(f"\n  로그 파일: {log_dir()}")
    bad = sum(1 for item in items if item.level == "bad")
    warn = sum(1 for item in items if item.level == "warn")
    _p("\n" + "-" * 62)
    if bad == 0:
        _p(f"  결과: 쓸 준비가 됐습니다. (주의 {warn}개)")
        _p("  부르는 말을 확인하려면:  .venv\\Scripts\\python davis.py --test-wake")
    else:
        _p(f"  결과: 꼭 고쳐야 할 문제 {bad}개, 주의 {warn}개")
        _p("  위 [문제] 항목을 먼저 해결해 주세요.")
    _p("-" * 62)
    _p("")
    return 1 if bad else 0


STARTUP_SHORTCUT_NAME = "자비스 시작.lnk"
OLD_STARTUP_SHORTCUT_NAMES = ("다비스 시작.lnk", "자비스 시작.lnk")


def startup_dir() -> Path | None:
    """윈도우 시작프로그램 폴더. 윈도우가 아니면 None."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _read_shortcut(path: Path) -> tuple[str, str]:
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        return "", ""
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($args[0]);"
        "Write-Output $s.TargetPath; Write-Output $s.Arguments"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script, str(path)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return "", ""
    lines = [line.strip() for line in (result.stdout or "").splitlines()]
    return (lines[0] if lines else ""), (lines[1] if len(lines) > 1 else "")


def _shortcut_vbs_path(target: str, arguments: str) -> Path | None:
    text = f"{arguments} {target}".strip()
    match = re.search(r'"?([^"]*\.vbs)"?', text, flags=re.I)
    if not match:
        return None
    return Path(match.group(1).strip())


def stale_startup_items(startup: Path) -> list[str]:
    messages: list[str] = []
    try:
        entries = sorted(startup.glob("*.lnk"))
    except OSError:
        return messages
    for path in entries:
        name = path.name
        if name not in OLD_STARTUP_SHORTCUT_NAMES and "자비스" not in name and "다비스" not in name:
            continue
        target, arguments = _read_shortcut(path)
        vbs = _shortcut_vbs_path(target, arguments)
        if vbs is None:
            messages.append(f"옛 시작 항목이 남아 있습니다: {name} (내용을 읽지 못했습니다. 직접 지워도 됩니다.)")
            continue
        try:
            exists = vbs.exists()
        except OSError:
            exists = False
        if not exists:
            messages.append(
                f"옛 시작 항목이 지워진 폴더를 가리킵니다: {name} → {vbs} "
                "(부팅 때 오류가 납니다. install.bat 을 다시 실행하면 정리됩니다.)"
            )
        elif name != STARTUP_SHORTCUT_NAME:
            messages.append(
                f"예전 자동 시작 항목이 남아 있습니다: {name} → {vbs} "
                "(그 폴더를 지운 뒤 install.bat 을 다시 실행하면 정리됩니다.)"
            )
    return messages


def cmd_test_voice(cfg: dict) -> int:
    _p("삐 소리를 내고 '네, 대표님'을 재생합니다...")
    beep()
    Speaker(cfg, log).say(str(cfg.get("wake_reply") or "네, 대표님"), cache=True)
    _p("들리면 정상입니다. 안 들리면 README 의 '문제 해결'을 보세요.")
    return 0


def cmd_test_wake(cfg: dict, seconds: float = 20.0) -> int:
    """마이크로 들으며 무엇이 들리는지, 깨우기가 되는지 보여 준다."""
    try:
        _need_numpy()
    except Exception as exc:
        _p(f"[문제] {exc}")
        return 1
    try:
        stream = MicStream(cfg, log).open()
    except MicError as exc:
        _p("[문제] 마이크를 열지 못했습니다.")
        for line in str(exc).splitlines():
            _p("   " + line)
        return 1

    _p("")
    words = ", ".join(f"'{w}'" for w in (cfg.get("wake_words") or []))
    _p(f"  마이크: {stream.device_name}")
    _p(f"  {seconds:.0f}초 동안 듣습니다. 부르는 말: {words}")
    _p("  3초쯤 뒤에 또박또박 '자비스' 라고 말해 보세요. (잘 안 되면 '헤이 자비스')")
    _p("")

    whisper = WhisperHolder()
    whisper.start(cfg, log)
    model = None
    try:
        model = VoskHolder().get(cfg, log)
    except Exception as exc:
        _p(f"  [주의] Vosk 를 쓰지 못합니다: {exc}")

    segmenter = SpeechSegmenter(cfg, log, end_silence_sec=0.55, max_sec=8.0, min_speech_sec=0.2)
    reader = MicReader(stream, log)
    reader.start()
    started = time.time()
    hits = 0
    try:
        while time.time() - started < seconds:
            pcm = reader.get(timeout=0.3)
            if pcm is None:
                continue
            for segment in segmenter.feed(pcm):
                _p(f"   [말소리] {segment.size / SAMPLE_RATE:.1f}초")
                vosk_hit, vosk_text = (None, "")
                if model is not None:
                    vosk_hit, vosk_text = vosk_listen_segment(model, cfg, segment)
                    if vosk_text:
                        _p(f"      Vosk 가 들은 말: {vosk_text}")
                text = ""
                try:
                    if whisper.is_ready():
                        text = transcribe_whisper(segment, cfg, whisper, log, timeout=5.0)
                except Exception:
                    text = ""
                if text:
                    _p(f"      Whisper 가 들은 말: {text}")
                hit = vosk_hit or (whisper_wake_hit(text, cfg) if text else None)
                if hit:
                    hits += 1
                    _p(f"      [깨우기 인식] '{hit}'  ← 정상입니다")
                else:
                    _p("      (깨우기 말 아님)")
    except Exception as exc:
        _p(f"[문제] {exc}")
    finally:
        reader.stop()
        stream.close()

    _p("")
    if hits:
        _p(f"  결과: {hits}번 알아들었습니다. 정상입니다.")
        _p("  실제로 켜 보려면 '자비스 시작.vbs' 를 더블클릭하세요.")
        return 0
    _p("  결과: 부르는 말을 알아듣지 못했습니다.")
    _p("  · 위에 '들은 말'이 아예 없으면 마이크 문제입니다: --list-mics 로 장치를 확인하고,")
    _p("    config.json 의 mic_device 에 그 이름(일부)을 적어 보세요.")
    _p("  · 들은 말이 엉뚱하면 마이크를 입 가까이(50cm 안쪽) 두고, mic_gain 을 2.0으로 올려 보세요.")
    _p("  · 주변이 시끄러우면 speech_threshold 를 0.02로 올려 보세요.")
    _p("  · 이 화면을 사진으로 찍어 버터에게 보내 주세요.")
    return 1


def cmd_list_mics(cfg: dict) -> int:
    _p("")
    try:
        for line in devices_report(cfg).splitlines():
            _p("  " + line)
    except Exception as exc:
        _p(f"  [문제] {exc}")
        return 1
    _p("")
    _p("  config.json 의 \"mic_device\" 에 위 이름의 일부(예: \"USB\")나 번호(예: 1)를 적으면 그 마이크를 씁니다.")
    _p("")
    return 0


def _acquire_single_instance(logger: logging.Logger) -> bool:
    """두 번 실행되어 마이크를 서로 잡는 일을 막는다.

    예전 판은 TCP 포트(47581)로 확인했는데, 윈도우에서 하이퍼V·WSL 이 그 포트를
    예약해 두면 '이미 실행 중'으로 잘못 판단해 아무것도 하지 않고 조용히 끝났다.
    여기서는 잠금 파일을 쓴다.
    """
    try:
        folder = log_dir()
        handle = (folder / "davis.lock").open("a+")
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        globals()["_SINGLETON_HANDLE"] = handle
        return True
    except Exception as exc:
        logger.info("이미 자비스가 실행 중입니다(잠금 파일을 잡을 수 없음: %s).", exc)
        return False


def _message_box(text: str, title: str = APP_TITLE, error: bool = False) -> None:
    """윈도우 알림창. 안 되면 화면에 출력."""
    try:
        if os.name == "nt":
            import ctypes
            flags = 0x10 if error else 0x40
            ctypes.windll.user32.MessageBoxW(None, text, title, flags)
            return
    except Exception:
        pass
    _p(text)


# ---------------------------------------------------------------------------
# 시작점
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="자비스 음성 비서 (가입 없이)")
    parser.add_argument("--console", action="store_true", help="로그를 화면에도 보여 주기")
    parser.add_argument("--window", action="store_true", help="상태 창을 띄우고 실행")
    parser.add_argument("--silent", action="store_true", help="창 없이 실행 (자동 시작용)")
    parser.add_argument("--setup", action="store_true", help="설정 파일 만들기 + 음성모델 내려받기")
    parser.add_argument("--model-path", default="", help="이미 받아 둔 음성모델 폴더 경로")
    parser.add_argument("--check", action="store_true", help="설치 상태 점검")
    parser.add_argument("--test-voice", action="store_true", help="스피커 소리 테스트")
    parser.add_argument("--test-wake", action="store_true", help="부르는 말 인식 테스트")
    parser.add_argument("--list-mics", action="store_true", help="마이크 목록 보기")
    parser.add_argument("--version", action="store_true", help="버전 보기")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.version:
        _p(f"{APP_NAME} {APP_VERSION}")
        return 0

    wants_screen = bool(args.console or args.check or args.setup or args.test_voice
                        or args.test_wake or args.list_mics)
    logger = setup_logging(console=wants_screen)
    logger.info("명령 실행: %s", " ".join(sys.argv[1:]) or "(기본: 실행)")
    logger.info("로그 파일: %s", log_dir())
    _install_global_handlers(logger)

    cfg = load_config()

    if args.setup:
        return cmd_setup(cfg, args.model_path or None)
    if args.check:
        return cmd_check(cfg)
    if args.test_voice:
        return cmd_test_voice(cfg)
    if args.test_wake:
        return cmd_test_wake(cfg)
    if args.list_mics:
        return cmd_list_mics(cfg)

    if not _acquire_single_instance(logger):
        message = (f"자비스가 이미 켜져 있습니다.\n\n"
                   f"작업 표시줄이나 숨은 아이콘에서 자비스 창을 찾아보세요.\n"
                   f"로그: {log_dir()}")
        logger.warning("이미 실행 중이라 새로 시작하지 않습니다.")
        if not args.silent:
            _message_box(message)
        return 0

    assistant = Assistant(cfg, logger)
    use_window = bool(cfg.get("show_window", True)) and not args.silent
    if args.window:
        use_window = True

    exit_code = 0
    if use_window:
        window = StatusWindow(assistant, cfg, logger, on_quit=assistant.stop)
        try:
            window._build()                      # 창을 먼저 만든다(실패하면 창 없이 실행)
            assistant.on_state = window.update_state
            assistant.on_heard = window.update_heard
            assistant.on_answer = window.update_answer
            assistant.speaker.on_state = window.update_state
        except Exception as exc:
            logger.warning("상태 창을 띄우지 못해 창 없이 실행합니다: %s", exc)
            window = None
        if window is not None:
            def worker() -> None:
                nonlocal exit_code
                try:
                    exit_code = assistant.run()
                except Exception as exc:
                    logger.exception("본체가 멈췄습니다: %s", exc)
                    exit_code = 1
                finally:
                    window.should_close = True   # 창 정리는 창을 만든 쪽(메인 스레드)이 한다

            thread = threading.Thread(target=worker, name="assistant", daemon=True)
            thread.start()
            try:
                window.run()
            except Exception as exc:
                logger.warning("상태 창이 멈춰 창 없이 이어갑니다: %s", exc)
            assistant.stop()
            thread.join(timeout=30)
            return exit_code

    try:
        return assistant.run()
    except KeyboardInterrupt:
        logger.info("사용자가 종료했습니다.")
        assistant.stop()
        return EXIT_USER_QUIT
    except Exception as exc:
        logger.exception("예상하지 못한 오류로 종료합니다: %s", exc)
        if not args.silent:
            _message_box(f"자비스가 오류로 멈췄습니다.\n\n{exc}\n\n로그: {log_dir()}", error=True)
        return 1
    finally:
        assistant.stop()


if __name__ == "__main__":
    sys.exit(main())
