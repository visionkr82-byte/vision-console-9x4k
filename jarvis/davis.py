#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자비스 — 윈도우 PC용 음성 비서 (v3 · 가입 없이)

"자비스"라고 부르면 "네, 대표님" 하고 대답한 뒤,
질문을 받아 클로드 코드(Claude Code CLI)로 답을 만들어 음성으로 읽어 줍니다.

부르는 말은 오프라인 음성인식(Vosk 한국어 모델)으로 알아듣습니다.
Picovoice 같은 가입이나 키는 필요 없습니다.

실행 방법
    pythonw davis.py               창 없이 실행 (자동 시작용)
    python  davis.py --console     로그를 보면서 실행
    python  davis.py --setup       설정 파일 만들기 + 한국어 음성모델 내려받기
    python  davis.py --check       설치 상태 점검
    python  davis.py --test-wake   부르는 말 인식 테스트 (20초)
    python  davis.py --test-voice  스피커 소리 테스트
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# pygame 이 콘솔에 홍보 문구를 찍지 않게 한다
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import numpy as np
except Exception:  # numpy 가 없으면 --check 가 알려 준다
    np = None  # type: ignore[assignment]

APP_NAME = "자비스"
APP_VERSION = "3.0.0"

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
EXAMPLE_CONFIG_NAME = "config.example.json"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")

SAMPLE_RATE = 16000          # Vosk 와 받아쓰기가 함께 쓰는 샘플레이트
BLOCK_MS = 30                # 한 번에 읽는 소리 길이

# 한국어 소형 모델 (약 87MB, 풀면 약 250MB)
VOSK_MODEL_NAME = "vosk-model-small-ko-0.22"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip"
VOSK_MODEL_MB = 87
MIN_ZIP_BYTES = 20 * 1024 * 1024   # 이보다 작으면 내려받기가 제대로 안 된 것으로 본다

DEFAULT_CONFIG = {
    # 부르는 말 (Vosk 한국어 모델)
    "vosk_model_path": "",
    "wake_words": ["자비스", "헤이 자비스"],
    "wake_sensitivity": 0.4,
    "wake_reply": "네, 대표님",
    # 받아쓰기 · 답변 · 목소리
    "whisper_model": "small",
    "tts_voice": "ko-KR-SunHiNeural",
    "max_answer_chars": 400,
    "claude_exe": "claude",
    "claude_timeout_sec": 180,
    "claude_cwd": "",
    # 마이크
    "silence_sec": 1.2,
    "max_record_sec": 10.0,
    "min_record_sec": 1.0,
    "listen_wait_sec": 8.0,
    "speech_threshold": 0.02,
    # 이름 · 안내
    "owner_name": "김용진",
    "assistant_name": "자비스",
    "announce_start": False,
}

# v1(Picovoice 판)에서 쓰던 항목. 남아 있으면 지워도 된다고 알려 준다.
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


def _local_app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / "davis-voice"
    return Path.home() / ".davis-voice"


def _log_path() -> Path:
    """로그는 프로그램 폴더에 쓰고, 안 되면 사용자 폴더에 쓴다."""
    try:
        probe = BASE_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return BASE_DIR / "davis.log"
    except Exception:
        d = _local_app_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d / "davis.log"


def _cache_dir() -> Path:
    d = _local_app_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _can_write(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def setup_logging(console: bool = False) -> logging.Logger:
    logger = logging.getLogger("davis")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    try:
        fh = logging.handlers.RotatingFileHandler(
            str(_log_path()), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    if console and sys.stdout is not None:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


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
            log.warning("config.json 을 읽지 못했습니다: %s", exc)
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
    # 프로그램 폴더 안에 이름이 조금 다른 모델 폴더가 있어도 찾아 준다
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

    request = urllib.request.Request(url, headers={"User-Agent": "davis-installer/3.0"})
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
    import tempfile as _tempfile

    dest_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(_tempfile.mkdtemp(prefix="davis_model_", dir=str(dest_parent)))
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
            # 기존 폴더는 지우지 않고 옆으로 치워 둔다
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
    zip_path = _cache_dir() / f"{VOSK_MODEL_NAME}.zip"
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


# ---------------------------------------------------------------------------
# 깨우기 말 — 문법 · 문자열 판정
# ---------------------------------------------------------------------------

# Vosk 가 '자비스'를 흐리게 들을 때 내놓는 다른 표기들.
# 이 모델의 어휘(graph/Gr.fst)에 실제로 있는 단어만 넣는다 — 없는 단어는 Vosk 가
# 그냥 버리고 경고만 찍는다(측정해서 확인함).
WAKE_STRONG_EXTRA = ("자비스",)

# 어휘에 없어서 문법에는 넣을 수 없지만, 들리면 깨운 것으로 보는 표기들
WAKE_MATCH_ONLY = ("자비수", "자비쓰", "자빗스", "짜비스")

# 앞 글자를 놓쳤을 때 나오는 형태. 말 전체가 아주 짧을 때만 인정한다.
# '자비'는 실제로 쓰는 말이라('자비를 베풀어') 넣지 않는다 — 넣으면 그 말에 깨운다(측정함).
WAKE_WEAK = ("비스",)

# 깨우기 말로 인정하려면 그 말이 실제로 소리 난 길이가 이만큼은 되어야 한다.
# 다른 말이 '자비스'로 우겨넣어진 경우는 대개 0.3초 안팎이었다(측정함).
MIN_WAKE_SEC = 0.45

# 문법에 함께 넣어 두는 흔한 말들.
# 이 말들이 있어야 Vosk 가 다른 말을 '자비스'로 우겨넣지 않는다.
# 실제로 '회의 자료 준비해 줘'는 '자료'가 없을 때 '회의 자비스 준비 해 줘'로,
# '자비를 베풀어 주세요'는 '자비'가 없을 때 '자비스 대표 줄래'로 들렸다(측정함).
DISTRACTOR_WORDS = (
    # '자비스'와 소리가 겹치는 말들
    "비스", "비스킷", "비스듬", "자비", "서비스", "디스", "디스크",
    "티켓", "로션", "베스트", "버스",
    # 인사·대답·되묻기
    "안녕", "고마워", "미안", "괜찮아", "좋아", "싫어", "맞아", "아니", "네", "예",
    "아니요", "그래", "그래요", "옳지", "네네", "아하", "잠깐", "잠시", "감사", "죄송",
    "뭐", "뭘", "뭐야", "뭐해", "왜", "왜요", "누구", "언제", "어디", "얼마", "몇", "며칠",
    "무슨", "어떤", "어떻게", "어때", "그거", "이거", "저거", "그것", "이것", "저것", "저기",
    # 사람·관계
    "나", "너", "우리", "저", "제", "내", "님", "사람", "사람들", "아들", "딸", "엄마", "아빠",
    "부모님", "친구", "선배", "후배", "대표", "사장", "회사", "사무실", "직원", "손님", "고객",
    "대표님", "사장님", "고객님", "팀장님", "직원분", "회사원", "거래처", "영업팀",
    # 시간
    "오늘", "어제", "내일", "요즘", "아침", "점심", "저녁", "밤", "새벽", "오전", "오후",
    "시간", "시", "분", "일찍", "지금", "먼저", "다음", "이전", "이번", "지난", "다음주",
    "이번주", "한달", "일년", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
    # 일·업무
    "약속", "회의", "회계", "자료", "보고서", "계약", "매출", "일정", "계획", "할일", "목록",
    "정리", "요약", "번역", "계산", "확인", "점검", "준비", "진행", "상태", "결과", "문제",
    "해결", "방법", "이유", "내용", "질문", "대답", "답변", "설명", "문서", "메일", "뉴스",
    "광고", "마케팅", "홍보", "노출", "순위", "키워드", "플레이스", "네이버", "블로그", "인스타그램",
    "유튜브", "카톡", "문자메시지",
    # 사물·생활
    "집", "차", "자동차", "택시", "지하철", "역", "버스", "학교", "병원", "은행", "마트", "편의점",
    "식당", "카페", "주차장", "공원", "밥", "물", "커피", "술", "맥주", "소주", "김치", "라면", "빵",
    "사과", "바나나", "음식", "한잔", "전화", "문자", "음악", "노래", "영상", "영화", "드라마", "사진",
    "파일", "폴더", "컴퓨터", "노트북", "핸드폰", "화면", "볼륨", "소리", "목소리", "알람", "예약",
    "취소", "배송", "주문", "결제", "가격", "할인", "쿠폰", "상품", "날씨", "비", "눈", "바람",
    "춥다", "미세먼지",
    # 움직임·부탁
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
    """Vosk 에 넘길 문법(인식할 단어 목록).

    이 모델의 어휘에 확실히 있는 단어만 넣는다. 없는 단어는 Vosk 가 버리고
    경고만 찍기 때문에, 뜻없이 넣으면 콘솔만 지저분해진다.
    """
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
    """단어별 시간이 있을 때: 깨우기 말이 '한 단어로, 충분히 길게' 들렸는지 본다.

    글자 일부만 겹치는 다른 단어('자비스트')나, 문장 속에서 아주 짧게
    우겨넣어진 조각은 깨우기 말로 보지 않는다.
    """
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
    """들은 말 안에서 깨우기 말을 찾는다. 찾으면 그 말을 돌려준다.

    speech_sec 는 그 말이 실제로 소리 난 길이(초). words 는 Vosk 가 준 단어별
    시작·끝 시간 목록이다. 단어 시간이 있으면 '비스'처럼 한 글자를 놓친 형태는
    말 전체가 아주 짧을 때만, 그리고 '자비스'는 그 말이 충분히 길게 들렸을 때만
    깨우기 말로 인정한다(문장 속에서 다른 말이 '자비스'로 우겨넣어지는 것을 막는다).
    """
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


@dataclass
class WakeHit:
    """깨우기 말을 들었을 때의 결과."""
    word: str
    text: str
    tail: Any = None          # 부르는 말 뒤에 이어 말한 소리(int16) 또는 None
    utterance: Any = None     # 이번에 한 말 전체의 소리(int16)
    speech_sec: float = 0.0


class WakeDetector:
    """마이크에서 온 30ms 덩어리를 받아 깨우기 말을 찾아낸다."""

    def __init__(self, model, cfg: dict, logger: logging.Logger = log,
                 samplerate: int = SAMPLE_RATE, block_ms: int = BLOCK_MS) -> None:
        np_mod = _need_numpy()
        self.np = np_mod
        self.cfg = cfg
        self.logger = logger
        self.sr = int(samplerate)
        self.block = int(self.sr * block_ms / 1000)
        self.rec = build_recognizer(model, cfg, logger)
        self.utt: list[Any] = []
        self.max_blocks = int(30 * 1000 / block_ms)   # 한 번에 30초까지만 들고 있는다
        self.resets = 0

    def feed(self, pcm) -> WakeHit | None:
        """소리 한 덩어리를 넣는다. 깨우기 말이면 WakeHit, 아니면 None."""
        np_mod = self.np
        pcm = np_mod.asarray(pcm, dtype=np_mod.int16).reshape(-1)
        if pcm.size == 0:
            return None
        self.utt.append(pcm)
        if len(self.utt) > self.max_blocks:
            # 30초가 넘게 한 번에 이어지면 앞부분을 버리고 새로 센다.
            # (여기서 일부만 잘라 두면 인식기와 시간이 어긋나므로 통째로 초기화한다)
            self.utt = [pcm]
            try:
                self.rec.Reset()
            except Exception:
                pass
            self.resets += 1
        try:
            finished = self.rec.AcceptWaveform(pcm.tobytes())
        except Exception as exc:
            self.logger.debug("깨우기 인식 오류: %s", exc)
            return None
        if not finished:
            return None
        try:
            result = json.loads(self.rec.Result())
        except Exception:
            result = {}
        hit = self.check(result)
        self.utt = [pcm]
        return hit

    def check(self, result: dict) -> WakeHit | None:
        text = str(result.get("text") or "").strip()
        if not text:
            return None
        words = result.get("result") or []
        speech_sec = 0.0
        if words:
            try:
                speech_sec = max(0.0, float(words[-1].get("end", 0.0)) - float(words[0].get("start", 0.0)))
            except (TypeError, ValueError):
                speech_sec = 0.0
        found = find_wake(text, self.cfg, speech_sec if words else None, words=words)
        if not found:
            return None

        # 깨운 말이 끝난 지점부터 뒤에 이어 말한 소리를 잘라 둔다
        wake_end = None
        target = squash(found)
        for item in words:
            if target and target in squash(item.get("word", "")):
                try:
                    wake_end = float(item.get("end", 0.0))
                except (TypeError, ValueError):
                    wake_end = None
                break
        if wake_end is None and words:
            try:
                wake_end = float(words[-1].get("end", 0.0))
            except (TypeError, ValueError):
                wake_end = None

        tail = None
        whole = None
        if self.utt:
            audio = self.np.concatenate(self.utt)
            whole = audio
            if wake_end is not None:
                start = int((wake_end + 0.08) * self.sr)
                if start < audio.size:
                    tail = audio[start:]
            else:
                tail = audio
        return WakeHit(word=found, text=text, tail=tail, utterance=whole, speech_sec=speech_sec)


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
        rec.SetWords(True)     # 단어별 시작·끝 시간을 받아야 이어 말한 질문을 잘라낼 수 있다
    except Exception:
        pass
    return rec


# ---------------------------------------------------------------------------
# 소리 (비프음 / 음성 출력)
# ---------------------------------------------------------------------------

def beep(logger: logging.Logger = log) -> None:
    """짧은 '삐' 소리. 외부 음원 파일 없이 numpy로 만든다."""
    try:
        np_mod = _need_numpy()
        import sounddevice as sd
        sr = 24000
        dur = 0.13
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


def _pygame_init() -> None:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()


def _play_mp3(path: Path, logger: logging.Logger = log) -> None:
    import pygame
    _pygame_init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    deadline = time.time() + 180
    while pygame.mixer.music.get_busy() and time.time() < deadline:
        time.sleep(0.05)
    try:
        pygame.mixer.music.unload()
    except Exception:
        pass


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
        # 도중에 끊겨 반쯤 쓰인 파일이 남으면 다음에 그대로 재생될 수 있다
        try:
            path.unlink()
        except Exception:
            pass
        raise


def speak_windows(text: str, logger: logging.Logger = log) -> None:
    """인터넷이나 edge-tts가 안 될 때 쓰는 윈도우 기본 음성."""
    text = clean_for_speech(text)
    if not text:
        return
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        logger.error("윈도우 기본 음성(powershell)을 찾지 못했습니다.")
        return
    txt_path = Path(tempfile.gettempdir()) / f"davis_tts_{os.getpid()}.txt"
    ps_path = Path(tempfile.gettempdir()) / f"davis_tts_{os.getpid()}.ps1"
    try:
        txt_path.write_text(text, encoding="utf-8")
        safe_path = str(txt_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech\r\n"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\r\n"
            "try { $s.SelectVoiceByHearingLocale('ko-KR') } catch {}\r\n"
            f"$t = Get-Content -Raw -Encoding UTF8 -LiteralPath '{safe_path}'\r\n"
            "$s.Speak($t)\r\n"
            "$s.Dispose()\r\n"
        )
        ps_path.write_text(script, encoding="utf-8-sig")
        subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
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


def speak(text: str, cfg: dict, logger: logging.Logger = log, cache: bool = False) -> None:
    """edge-tts 로 mp3를 만들어 재생한다. 실패하면 윈도우 기본 음성으로 대체."""
    text = clean_for_speech(text)
    if not text:
        return
    voice = str(cfg.get("tts_voice") or DEFAULT_CONFIG["tts_voice"])
    tmp_path = None
    try:
        if cache:
            import hashlib
            digest = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()[:16]
            mp3_path = _cache_dir() / f"{digest}.mp3"
        else:
            fd, name = tempfile.mkstemp(prefix="davis_ans_", suffix=".mp3")
            os.close(fd)
            mp3_path = Path(name)
            tmp_path = mp3_path

        if not mp3_path.exists() or mp3_path.stat().st_size < 400:
            _edge_tts_save(text, voice, mp3_path)
        _play_mp3(mp3_path, logger)
        return
    except Exception as exc:
        logger.warning("edge-tts 재생 실패(%s) → 윈도우 기본 음성으로 대체", exc)
        speak_windows(text, logger)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 받아쓰기 (faster-whisper → 안 되면 Vosk)
# ---------------------------------------------------------------------------

class WhisperNotReady(Exception):
    pass


class WhisperHolder:
    """모델 로딩은 백그라운드로. 첫 질문이 왔을 때 준비 안 됐으면 잠깐만 기다린다."""

    def __init__(self) -> None:
        self.model = None
        self.ready = threading.Event()
        self.error: Exception | None = None
        self._lock = threading.Lock()
        self._loading = False
        self._last_try = 0.0

    def start(self, cfg: dict, logger: logging.Logger = log) -> None:
        self._kick(cfg, logger)

    def _kick(self, cfg: dict, logger: logging.Logger = log) -> None:
        with self._lock:
            if self._loading:
                return
            self._loading = True
            self.ready.clear()
        self._last_try = time.time()
        threading.Thread(target=self._load, args=(cfg, logger), daemon=True).start()

    def _load(self, cfg: dict, logger: logging.Logger = log) -> None:
        try:
            from faster_whisper import WhisperModel
            logger.info("받아쓰기 모델(%s)을 준비합니다.", cfg.get("whisper_model"))
            started = time.time()
            model = WhisperModel(
                str(cfg.get("whisper_model") or "small"),
                device="cpu",
                compute_type="int8",
            )
            with self._lock:
                self.model = model
                self.error = None
            self.ready.set()
            logger.info("받아쓰기 모델 준비 완료 (%.1f초)", time.time() - started)
        except Exception as exc:
            with self._lock:
                self.error = exc
            self.ready.set()  # 실패해도 다음 질문이 오래 기다리지 않게
            logger.warning("받아쓰기 모델 준비 실패(Vosk 로 대신합니다): %s", exc)
        finally:
            with self._lock:
                self._loading = False

    def get(self, cfg: dict, logger: logging.Logger = log, timeout: float = 25.0):
        if self.model is not None:
            return self.model
        if self.ready.wait(timeout) and self.model is not None:
            return self.model
        # 한 번 실패했으면 3분에 한 번만 다시 시도한다 (인터넷이 돌아온 경우 등)
        if self.error is not None and not self._loading and (time.time() - self._last_try) > 180:
            self._kick(cfg, logger)
        return None


_WHISPER_NOISE = {
    "시청해주셔서 감사합니다.",
    "시청해 주셔서 감사합니다.",
    "구독과 좋아요 부탁드립니다.",
    "감사합니다.",
    "고맙습니다.",
    "음",
    "어",
}


def transcribe_whisper(audio, cfg: dict, holder: WhisperHolder, logger: logging.Logger = log) -> str:
    model = holder.get(cfg, logger)
    if model is None:
        raise WhisperNotReady("받아쓰기 모델이 아직 준비되지 않았습니다.")
    segments, _info = model.transcribe(
        audio,
        language="ko",
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'").strip()
    if text in _WHISPER_NOISE or len(text) < 2:
        return ""
    return text


def transcribe_vosk(audio, holder: VoskHolder, cfg: dict, logger: logging.Logger = log) -> str:
    """Vosk 전체 모델로 받아쓴다(Whisper 를 못 쓸 때)."""
    np_mod = _need_numpy()
    vosk = _import_vosk(logger)
    model = holder.get(cfg, logger)
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    pcm = np_mod.asarray(audio, dtype=np_mod.float32)
    pcm = np_mod.clip(pcm, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np_mod.int16)
    step = int(SAMPLE_RATE * BLOCK_MS / 1000)
    texts: list[str] = []
    for start in range(0, max(1, pcm.size), step):
        chunk = pcm[start:start + step]
        if chunk.size == 0:
            continue
        if rec.AcceptWaveform(chunk.tobytes()):
            try:
                texts.append(str(json.loads(rec.Result()).get("text") or ""))
            except Exception:
                pass
    try:
        texts.append(str(json.loads(rec.FinalResult()).get("text") or ""))
    except Exception:
        pass
    text = re.sub(r"\s+", " ", " ".join(t for t in texts if t)).strip()
    if len(text) < 2:
        return ""
    return text


def transcribe(audio, cfg: dict, whisper: WhisperHolder, vosk: VoskHolder,
               logger: logging.Logger = log) -> str:
    """질문 받아쓰기. Whisper(품질)를 먼저 쓰고, 안 되면 Vosk 로 받아쓴다."""
    text = ""
    try:
        text = transcribe_whisper(audio, cfg, whisper, logger)
    except WhisperNotReady:
        text = ""
    except Exception as exc:
        logger.warning("Whisper 받아쓰기 실패: %s", exc)
        text = ""
    if text:
        return text
    try:
        fallback = transcribe_vosk(audio, vosk, cfg, logger)
        if fallback:
            logger.info("Vosk 로 받아썼습니다: %d자", len(fallback))
        return fallback
    except Exception as exc:
        logger.warning("Vosk 받아쓰기도 실패: %s", exc)
    if whisper.get(cfg, logger, timeout=0.1) is None:
        raise WhisperNotReady("받아쓰기 모델이 아직 준비되지 않았습니다.")
    return ""


# ---------------------------------------------------------------------------
# 마이크
# ---------------------------------------------------------------------------

class MicError(Exception):
    pass


def _pcm_array(data):
    np_mod = _need_numpy()
    if isinstance(data, np_mod.ndarray):
        return np_mod.asarray(data, dtype=np_mod.int16).reshape(-1)
    return np_mod.frombuffer(data, dtype=np_mod.int16)


def _rms(pcm) -> float:
    np_mod = _need_numpy()
    if pcm.size == 0:
        return 0.0
    x = pcm.astype(np_mod.float32) / 32768.0
    return float(np_mod.sqrt(float(np_mod.mean(np_mod.square(x)))))


def open_input_stream(block_samples: int):
    """16kHz 마이크를 연다. 그 샘플레이트가 안 되면 기본 레이트로 열고 바꿔 준다."""
    _need_numpy()
    import sounddevice as sd

    def _open(rate: int):
        stream = sd.RawInputStream(
            samplerate=rate, blocksize=max(1, int(block_samples * rate / SAMPLE_RATE)),
            dtype="int16", channels=1,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            raise
        return stream, rate

    try:
        return _open(SAMPLE_RATE)
    except Exception:
        try:
            default_rate = int(sd.query_devices(kind="input").get("default_samplerate") or 48000)
        except Exception:
            default_rate = 48000
        return _open(default_rate)


def resample_to_16k(pcm, from_rate: int):
    """다른 샘플레이트로 받은 소리를 16kHz 로 바꾼다."""
    np_mod = _need_numpy()
    pcm = np_mod.asarray(pcm, dtype=np_mod.int16).reshape(-1)
    if from_rate == SAMPLE_RATE or pcm.size == 0:
        return pcm
    n = int(round(pcm.size * SAMPLE_RATE / float(from_rate)))
    if n <= 0:
        return np_mod.zeros(0, dtype=np_mod.int16)
    x = pcm.astype(np_mod.float32)
    idx = np_mod.linspace(0, pcm.size - 1, n)
    out = np_mod.interp(idx, np_mod.arange(pcm.size), x)
    return np_mod.clip(np_mod.round(out), -32768, 32767).astype(np_mod.int16)


def record_question(cfg: dict, logger: logging.Logger = log):
    """질문을 녹음해 float32(-1~1) 배열로 돌려준다. 말이 없으면 None."""
    try:
        np_mod = _need_numpy()
        import sounddevice  # noqa: F401  (설치 여부만 확인한다)
    except Exception as exc:
        raise MicError(f"오디오 라이브러리를 불러오지 못했습니다: {exc}")

    block = int(SAMPLE_RATE * BLOCK_MS / 1000)          # 480 샘플
    silence_blocks = max(1, int(float(cfg["silence_sec"]) * 1000 / BLOCK_MS))
    min_blocks = max(1, int(float(cfg["min_record_sec"]) * 1000 / BLOCK_MS))
    max_blocks = max(min_blocks, int(float(cfg["max_record_sec"]) * 1000 / BLOCK_MS))
    wait_blocks = max(1, int(float(cfg["listen_wait_sec"]) * 1000 / BLOCK_MS))
    threshold = float(cfg["speech_threshold"] or 0.02)

    frames = []
    stream = None
    try:
        stream, rate = open_input_stream(block)
        # 주변 소음 측정 → 그보다 확실히 큰 소리를 말소리로 본다
        ambient = []
        for _ in range(8):
            data, _overflow = stream.read(block)
            ambient.append(_rms(resample_to_16k(_pcm_array(data), rate)))
        ambient.sort()
        noise = ambient[len(ambient) // 2]
        level = max(threshold, noise * 3.0)

        started = False
        for _ in range(wait_blocks):
            data, _overflow = stream.read(block)
            pcm = resample_to_16k(_pcm_array(data), rate)
            if _rms(pcm) > level:
                frames.append(pcm)
                started = True
                break
        if not started:
            logger.info("말소리를 듣지 못했습니다.")
            return None

        silence = 0
        while len(frames) < max_blocks:
            data, _overflow = stream.read(block)
            pcm = resample_to_16k(_pcm_array(data), rate)
            frames.append(pcm)
            if _rms(pcm) <= level:
                silence += 1
            else:
                silence = 0
            if len(frames) >= min_blocks and silence >= silence_blocks:
                break
    except MicError:
        raise
    except Exception as exc:
        raise MicError(str(exc))
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    if not frames:
        return None
    audio = np_mod.concatenate(frames)
    if audio.size < int(SAMPLE_RATE * 0.4):
        return None
    logger.debug("녹음 %.1f초", audio.size / SAMPLE_RATE)
    return audio.astype(np_mod.float32) / 32768.0


def trim_silence(audio, threshold: float = 0.02):
    """앞뒤 무음을 잘라 낸다(float32 배열)."""
    np_mod = _need_numpy()
    x = np_mod.asarray(audio, dtype=np_mod.float32).reshape(-1)
    if x.size == 0:
        return x
    frame = int(SAMPLE_RATE * 0.03)
    frames = max(1, x.size // frame)
    energy = []
    for i in range(frames):
        chunk = x[i * frame:(i + 1) * frame]
        energy.append(_rms((chunk * 32767.0).astype(np_mod.int16)))
    level = max(threshold, min(energy) if energy else threshold)
    loud = [i for i, value in enumerate(energy) if value > level * 1.5]
    if not loud:
        return x[:0]
    start = max(0, loud[0] - 2)
    end = min(frames, loud[-1] + 3)
    return x[start * frame:end * frame]


def speech_seconds(audio, threshold: float = 0.02) -> float:
    """소리 배열에서 실제로 말이 있는 길이(초)를 어림한다."""
    np_mod = _need_numpy()
    x = np_mod.asarray(audio, dtype=np_mod.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    frame = int(SAMPLE_RATE * 0.03)
    count = 0
    for i in range(max(1, x.size // frame)):
        chunk = x[i * frame:(i + 1) * frame]
        if _rms((chunk * 32767.0).astype(np_mod.int16)) > max(threshold, 0.015):
            count += 1
    return count * frame / float(SAMPLE_RATE)


def wait_for_wake(cfg: dict, vosk_holder: VoskHolder, logger: logging.Logger = log) -> WakeHit:
    """깨우기 말이 들릴 때까지 마이크를 듣는다. 장치 오류는 MicError로 넘긴다."""
    try:
        _need_numpy()
    except Exception as exc:
        raise MicError(str(exc))
    model = vosk_holder.get(cfg, logger)
    detector = WakeDetector(model, cfg, logger)
    block = detector.block
    stream = None
    try:
        stream, rate = open_input_stream(block)
        logger.debug("깨우기 대기 중 (마이크 %dHz)", rate)
        while True:
            data, _overflow = stream.read(block)
            pcm = _pcm_array(data)
            if rate != SAMPLE_RATE:
                pcm = resample_to_16k(pcm, rate)
            hit = detector.feed(pcm)
            if hit is not None:
                return hit
    except ModelError:
        raise
    except Exception as exc:
        raise MicError(str(exc))
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


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


def resolve_claude(name: str) -> str | None:
    name = str(name or "claude").strip() or "claude"
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "claude.exe",
        home / ".local" / "bin" / "claude.cmd",
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


def claude_command(exe: str, prompt: str) -> list[str]:
    return [exe, "-p", prompt, "--output-format", "text"]


def ask_claude(question: str, cfg: dict, logger: logging.Logger = log) -> str:
    exe = resolve_claude(str(cfg.get("claude_exe") or "claude"))
    if not exe:
        raise ClaudeNotFound("claude 실행 파일을 찾지 못했습니다.")

    prompt = build_prompt(question, cfg)
    cwd_text = str(cfg.get("claude_cwd") or "").strip()
    cwd = str(_expand(cwd_text)) if cwd_text else str(Path.home())
    try:
        timeout = float(cfg.get("claude_timeout_sec") or 180)
    except (TypeError, ValueError):
        timeout = 180.0

    cmd = claude_command(exe, prompt)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        raise ClaudeTimeout(f"{timeout}초 안에 답이 오지 않았습니다.")
    except OSError as exc:
        raise ClaudeNotFound(str(exc))

    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0 or not out:
        merged = (err + " " + out).lower()
        auth_hints = ("login", "log in", "logged in", "auth", "api key", "credential", "oauth", "unauthorized")
        if any(hint in merged for hint in auth_hints):
            raise ClaudeAuthError(err[:300] or "로그인이 필요합니다.")
        raise ClaudeError(err[:300] or f"claude 종료 코드 {result.returncode}")
    return out


# ---------------------------------------------------------------------------
# 설정 마법사 (--setup)
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


def download_model_with_progress(cfg: dict, logger: logging.Logger = log) -> Path:
    _p(f"   한국어 음성모델을 내려받습니다. 약 {VOSK_MODEL_MB}MB, 1~3분 걸립니다.")
    _p("   (한 번만 받으면 됩니다. 창이 멈춘 것처럼 보여도 기다려 주세요.)")
    path = ensure_model(cfg, logger, on_progress=_progress_printer())
    _p("")
    return path


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

    try:
        dest = download_model_with_progress(cfg)
    except Exception as exc:
        _p("")
        _p(f"   [문제] 음성모델을 자동으로 받지 못했습니다: {exc}")
        _p("")
        _p("   인터넷이 되는 브라우저에서 아래 주소를 열어 파일을 받으세요.")
        _p(f"   {VOSK_MODEL_URL}")
        _p("")
        _p("   받은 압축 파일을 풀면 'vosk-model-small-ko-0.22' 폴더가 나옵니다.")
        _p("   그 폴더를 이 프로그램 폴더에 넣고 다시 실행하면 됩니다.")
        _p("   다른 곳에 두었다면 아래처럼 위치를 알려 줘도 됩니다.")
        _p('   .venv\\Scripts\\python davis.py --setup --model-path "D:\\vosk-model-small-ko-0.22"')
        _p("")
        return 1

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


# ---------------------------------------------------------------------------
# 상태 점검 (--check)
# ---------------------------------------------------------------------------

STARTUP_SHORTCUT_NAME = "자비스 시작.lnk"
# 예전 판이 시작프로그램에 만들어 둔 이름들. 새 설치가 자기 것으로 다시 걸고 지운다.
OLD_STARTUP_SHORTCUT_NAMES = ("다비스 시작.lnk", "자비스 시작.lnk")


def _dep_status(module_name: str, label: str) -> tuple[bool, str]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "")
        return True, f"{label} {version}".strip()
    except Exception as exc:
        return False, f"{label} 없음/오류 ({exc.__class__.__name__})"


def startup_dir() -> Path | None:
    """윈도우 시작프로그램 폴더. 윈도우가 아니면 None."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _read_shortcut(path: Path) -> tuple[str, str]:
    """바로가기(.lnk)의 대상과 인자를 읽는다. 윈도우 PowerShell 로만 된다."""
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
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return "", ""
    lines = [line.strip() for line in (result.stdout or "").splitlines()]
    target = lines[0] if lines else ""
    arguments = lines[1] if len(lines) > 1 else ""
    return target, arguments


def _shortcut_vbs_path(target: str, arguments: str) -> Path | None:
    """바로가기가 가리키는 .vbs 파일. 찾지 못하면 None."""
    text = f"{arguments} {target}".strip()
    match = re.search(r'"?([^"]*\.vbs)"?', text, flags=re.I)
    if not match:
        return None
    return Path(match.group(1).strip())


def stale_startup_items(startup: Path) -> list[str]:
    """시작프로그램 폴더에서 손봐야 할 자비스 항목을 찾아 안내 문구로 돌려준다.

    예전 판의 '다비스 시작' 바로가기가 지워진 폴더를 가리키고 있으면 부팅 때
    오류 창이 뜬다. install.bat 을 다시 실행하면 새 것으로 바꾸면서 옛것을 지운다.
    """
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
            messages.append(
                f"옛 시작 항목이 남아 있습니다: {name} (내용을 읽지 못했습니다. 직접 지워도 됩니다.)"
            )
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


def cmd_check(cfg: dict) -> int:
    problems = 0
    warns = 0

    def ok(msg: str) -> None:
        _p(f"  [정상] {msg}")

    def warn(msg: str) -> None:
        nonlocal warns
        warns += 1
        _p(f"  [주의] {msg}")

    def bad(msg: str) -> None:
        nonlocal problems
        problems += 1
        _p(f"  [문제] {msg}")

    _p("")
    _p("=" * 60)
    _p(f"  자비스 상태 점검 (v{APP_VERSION} · 가입 없이)")
    _p("=" * 60)

    # 1. 설정 파일
    _p("\n[1] 설정 파일")
    if CONFIG_PATH.exists():
        ok(f"config.json 있음 ({CONFIG_PATH})")
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            legacy = [k for k in raw if k in LEGACY_CONFIG_KEYS]
            if legacy:
                warn("예전 항목이 남아 있습니다(써도 지장 없음): " + ", ".join(legacy))
        except Exception:
            warn("config.json 을 읽지 못했습니다. install.bat 을 다시 실행해 주세요.")
    else:
        bad("config.json 이 없습니다. install.bat 을 다시 실행하거나 --setup 을 실행하세요.")
    words = ", ".join(f"'{w}'" for w in (cfg.get("wake_words") or []))
    ok(f"부르는 말: {words or '(없음)'}")
    level = _sensitivity(cfg)
    if 0.2 <= level <= 1.0:
        ok(f"깨우기 민감도: {level:.2f}")
    else:
        warn(f"wake_sensitivity 값({level:.2f})이 범위를 벗어났습니다. 0.4를 권합니다.")

    # 2. 한국어 음성모델
    _p("\n[2] 한국어 음성모델 (부르는 말을 알아듣는 데 씁니다)")
    model_dir = find_model_dir(cfg)
    if model_dir is None:
        bad("음성모델이 없습니다. install.bat 을 다시 실행하거나 --setup 을 실행하세요.")
    else:
        ok(f"모델 폴더: {model_dir} ({model_size_mb(model_dir):.0f}MB)")
        try:
            vosk = _import_vosk()
            started = time.time()
            model = vosk.Model(str(model_dir))
            ok(f"모델을 읽었습니다 ({time.time() - started:.1f}초)")
            try:
                recognizer = build_recognizer(model, cfg)
                del recognizer
                ok("깨우기 문법을 만들었습니다: " + words)
            except Exception as exc:
                bad(f"깨우기 문법을 만들지 못했습니다: {exc}")
            del model
        except Exception as exc:
            bad(f"모델을 읽지 못했습니다: {exc}")

    # 3. 파이썬 패키지
    _p("\n[3] 필요한 프로그램")
    for module_name, label in (
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("vosk", "vosk"),
        ("faster_whisper", "faster-whisper"),
        ("edge_tts", "edge-tts"),
        ("pygame", "pygame"),
    ):
        good, text = _dep_status(module_name, label)
        (ok if good else bad)(text)

    # 4. 마이크/스피커
    _p("\n[4] 마이크와 스피커")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if int(d.get("max_input_channels", 0)) > 0]
        outputs = [d for d in devices if int(d.get("max_output_channels", 0)) > 0]
        if inputs:
            try:
                default_in = sd.query_devices(kind="input")
                ok(f"마이크 {len(inputs)}개 / 기본 입력: {default_in.get('name', '알 수 없음')}")
            except Exception:
                ok(f"마이크 {len(inputs)}개")
        else:
            bad("입력 장치(마이크)를 찾지 못했습니다.")
        if outputs:
            ok(f"스피커 {len(outputs)}개")
        else:
            warn("출력 장치(스피커)를 찾지 못했습니다.")
    except Exception as exc:
        bad(f"오디오 장치를 확인하지 못했습니다: {exc}")

    # 5. 클로드 코드
    _p("\n[5] 클로드 코드 (Claude Code CLI)")
    exe = resolve_claude(str(cfg.get("claude_exe") or "claude"))
    if not exe:
        bad("claude 명령을 찾지 못했습니다. Node.js 설치 후 'npm install -g @anthropic-ai/claude-code' 가 필요합니다.")
    else:
        try:
            result = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                creationflags=CREATE_NO_WINDOW,
            )
            version_text = (result.stdout or result.stderr or "").strip().splitlines()
            ok(f"claude 찾음: {exe}" + (f" ({version_text[0]})" if version_text else ""))
            ok("로그인 여부는 실제 질문을 해 봐야 확인됩니다. 안 되면 cmd 창에서 'claude' 를 실행해 로그인하세요.")
        except Exception as exc:
            warn(f"claude 는 찾았지만 실행 확인 실패: {exc}")

    # 6. 음성 출력
    _p("\n[6] 음성 출력")
    try:
        text = "자비스 점검 중입니다."
        fd, name = tempfile.mkstemp(prefix="davis_check_", suffix=".mp3")
        os.close(fd)
        path = Path(name)
        _edge_tts_save(text, str(cfg.get("tts_voice") or DEFAULT_CONFIG["tts_voice"]), path)
        ok("인터넷 음성(edge-tts) 사용 가능")
        try:
            path.unlink()
        except Exception:
            pass
    except Exception as exc:
        warn(f"인터넷 음성(edge-tts) 실패: {exc} → 윈도우 기본 음성으로 대체됩니다.")

    # 7. 자동 시작
    _p("\n[7] 컴퓨터 켤 때 자동 시작")
    try:
        startup = startup_dir()
        if startup is None:
            warn("시작프로그램 폴더를 찾지 못했습니다(윈도우가 아닌 것 같습니다).")
        else:
            shortcut = startup / STARTUP_SHORTCUT_NAME
            if shortcut.exists():
                ok(f"자동 시작 바로가기 있음 ({shortcut})")
            else:
                warn("자동 시작 바로가기가 없습니다. install.bat 을 다시 실행하면 만들어집니다.")
            for item in stale_startup_items(startup):
                warn(item)
    except Exception as exc:
        warn(f"자동 시작 항목을 확인하지 못했습니다: {exc}")

    _p("\n" + "-" * 60)
    if problems == 0:
        _p(f"  결과: 쓸 준비가 됐습니다. (주의 {warns}개)")
        _p("  소리를 확인하려면:  .venv\\Scripts\\python davis.py --test-voice")
        _p("  부르는 말을 확인하려면:  .venv\\Scripts\\python davis.py --test-wake")
    else:
        _p(f"  결과: 꼭 고쳐야 할 문제 {problems}개, 주의 {warns}개")
        _p("  위 [문제] 항목을 먼저 해결해 주세요.")
    _p("-" * 60)
    _p("")
    return 1 if problems else 0


def cmd_test_voice(cfg: dict) -> int:
    _p("삐 소리를 내고 '네, 대표님'을 재생합니다...")
    beep()
    speak(str(cfg.get("wake_reply") or "네, 대표님"), cfg, cache=True)
    _p("들리면 정상입니다. 안 들리면 README 의 '문제 해결'을 보세요.")
    return 0


def cmd_test_wake(cfg: dict, seconds: float = 20.0) -> int:
    """20초 동안 마이크를 들으며 무엇이 들리는지, 깨우기가 되는지 보여 준다."""
    try:
        _need_numpy()
    except Exception as exc:
        _p(f"[문제] {exc}")
        return 1
    holder = VoskHolder()
    try:
        model = holder.get(cfg)
    except Exception as exc:
        _p(f"[문제] {exc}")
        _p("먼저 install.bat 또는 'davis.py --setup' 으로 음성모델을 준비해 주세요.")
        return 1

    vosk = _import_vosk()
    detector = WakeDetector(model, cfg)
    full = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    block = detector.block

    words = ", ".join(f"'{w}'" for w in (cfg.get("wake_words") or []))
    _p("")
    _p(f"  {seconds:.0f}초 동안 마이크로 듣습니다. 부르는 말: {words}")
    _p("  3초쯤 뒤에 또박또박 '자비스' 라고 말해 보세요. (잘 안 되면 '헤이 자비스')")
    _p("")
    last_partial = ""
    started = time.time()
    hits = 0
    stream = None
    try:
        stream, rate = open_input_stream(block)
        while time.time() - started < seconds:
            data, _overflow = stream.read(block)
            pcm = _pcm_array(data)
            if rate != SAMPLE_RATE:
                pcm = resample_to_16k(pcm, rate)
            try:
                if full.AcceptWaveform(pcm.tobytes()):
                    text = str(json.loads(full.Result()).get("text") or "").strip()
                    if text:
                        _p(f"   들린 말: {text}")
                        last_partial = ""
                else:
                    text = str(json.loads(full.PartialResult()).get("partial") or "").strip()
                    if text and text != last_partial:
                        last_partial = text
            except Exception:
                pass
            hit = detector.feed(pcm)
            if hit is not None:
                hits += 1
                _p(f"   [깨우기 인식] '{hit.word}'  (들은 말: {hit.text})")
    except Exception as exc:
        _p(f"[문제] 마이크를 열지 못했습니다: {exc}")
        return 1
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    _p("")
    if hits:
        _p(f"  결과: {hits}번 알아들었습니다. 정상입니다.")
        _p("  실제로 켜 보려면 '자비스 시작.vbs' 를 더블클릭하세요.")
        return 0
    _p("  결과: 부르는 말을 알아듣지 못했습니다.")
    _p("  · 조용한 곳에서, 마이크 가까이에서 또박또박 말해 보세요.")
    _p("  · '헤이 자비스' 라고 길게 불러 보세요.")
    _p("  · 마이크가 기본 입력 장치로 잡혀 있는지 윈도우 설정 → 시스템 → 소리에서 확인하세요.")
    _p("  · config.json 의 wake_sensitivity 를 0.8 로 올려 보세요.")
    return 1


# ---------------------------------------------------------------------------
# 본 동작
# ---------------------------------------------------------------------------

def _speak_notice(text: str, cfg: dict, logger: logging.Logger) -> None:
    logger.info("안내: %s", text)
    speak(text, cfg, logger, cache=True)


def _looks_like_wake_token(token: str, forms: set[str]) -> bool:
    """받아쓴 첫 단어가 깨우기 말 비슷한지( '바디스' 처럼 한두 글자 틀린 경우)."""
    t = squash(token)
    if not t:
        return False
    if t in forms:
        return True
    if len(t) == 3 and (t.endswith("스") or t.endswith("쓰")) and _char_distance(t, squash("자비스")) <= 2:
        return True
    return False


def strip_wake_prefix(text: str, cfg: dict) -> str:
    """받아쓴 문장 앞에 붙은 깨우기 말('자비스')을 지운다."""
    s = str(text or "").strip()
    if not s:
        return ""
    forms = {squash(w) for w in wake_words(cfg)} | {squash(w) for w in WAKE_WEAK}
    forms.discard("")
    tokens = s.split()
    for count in (3, 2, 1):
        if count > len(tokens):
            continue
        head = " ".join(tokens[:count])
        if squash(head) in forms or (count > 1 and _looks_like_wake_token(head, forms)):
            return " ".join(tokens[count:]).strip(" ,.!?~")
    if tokens and _looks_like_wake_token(tokens[0], forms):
        return " ".join(tokens[1:]).strip(" ,.!?~")
    # 띄어쓰기 없이 붙어 있는 경우 (두 글자 짜리는 다른 단어의 앞부분일 수 있어 보지 않는다)
    squashed = squash(s)
    for form in sorted((f for f in forms if len(f) >= 3), key=len, reverse=True):
        if squashed.startswith(form) and len(squashed) > len(form):
            cut = 0
            seen = 0
            while cut < len(s) and seen < len(form):
                if squash(s[cut]):
                    seen += 1
                cut += 1
            return s[cut:].strip(" ,.!?~")
    return s


def _tail_question(hit: WakeHit | None, cfg: dict, logger: logging.Logger):
    """부르면서 이어 말한 소리가 있으면 (말 전체 소리, True)를 돌려준다.

    깨우기 말이 끝나는 지점을 잘라내면 다음 단어 첫소리가 날아가므로,
    말 전체를 받아쓰고 앞의 '자비스'만 지우는 방식을 쓴다.
    """
    if hit is None or hit.utterance is None:
        return None, False
    try:
        np_mod = _need_numpy()
        threshold = float(cfg.get("speech_threshold") or 0.02)
        tail = hit.tail
        if tail is None:
            return None, False
        tail_audio = np_mod.asarray(tail, dtype=np_mod.int16).reshape(-1).astype(np_mod.float32) / 32768.0
        if speech_seconds(tail_audio, threshold) < 0.35:
            return None, False
        whole = np_mod.asarray(hit.utterance, dtype=np_mod.int16).reshape(-1).astype(np_mod.float32) / 32768.0
        whole = trim_silence(whole, threshold)
        if whole.size < int(SAMPLE_RATE * 0.4):
            return None, False
        logger.info("부르면서 이어 말한 질문을 씁니다 (%.1f초).", whole.size / SAMPLE_RATE)
        return whole, True
    except Exception as exc:
        logger.debug("이어 말한 소리를 쓰지 못했습니다: %s", exc)
        return None, False


def _interaction(cfg: dict, whisper: WhisperHolder, vosk_holder: VoskHolder,
                 logger: logging.Logger, hit: WakeHit | None = None) -> None:
    """깨우기 말이 울린 뒤 한 번의 대화."""
    beep(logger)
    speak(str(cfg.get("wake_reply") or "네, 대표님"), cfg, logger, cache=True)

    audio, from_wake = _tail_question(hit, cfg, logger)
    if audio is None:
        try:
            audio = record_question(cfg, logger)
        except MicError as exc:
            logger.warning("마이크 오류: %s", exc)
            _speak_notice("마이크를 찾지 못했어요. 마이크가 꽂혀 있는지 확인해 주세요.", cfg, logger)
            return

    if audio is None:
        _speak_notice("듣지 못했어요. 다시 불러 주세요.", cfg, logger)
        return

    started = time.time()
    try:
        question = transcribe(audio, cfg, whisper, vosk_holder, logger)
    except WhisperNotReady:
        logger.warning("받아쓰기 모델 준비 안 됨")
        _speak_notice(
            "받아쓰기 준비 중이에요. 처음 한 번은 파일을 받느라 몇 분 걸립니다. 잠시 뒤 다시 불러 주세요.",
            cfg,
            logger,
        )
        return
    except Exception as exc:
        logger.exception("받아쓰기 실패: %s", exc)
        _speak_notice("말을 글로 바꾸는 데 실패했어요. 다시 말씀해 주세요.", cfg, logger)
        return
    if not question:
        _speak_notice("잘 못 들었어요. 다시 말씀해 주세요.", cfg, logger)
        return
    if from_wake:
        stripped = strip_wake_prefix(question, cfg)
        if stripped != question:
            logger.info("앞의 부르는 말을 지웠습니다: %d자 → %d자", len(question), len(stripped))
            question = stripped
        if len(question) < 2:
            _speak_notice("듣지 못했어요. 다시 불러 주세요.", cfg, logger)
            return
    logger.info("질문 인식 완료 (%d자, %.1f초)", len(question), time.time() - started)

    started = time.time()
    try:
        answer = ask_claude(question, cfg, logger)
    except ClaudeNotFound:
        logger.error("claude 실행 파일을 찾지 못했습니다.")
        _speak_notice(
            "클로드 코드가 설치되지 않았어요. 설치 안내를 확인해 주세요.", cfg, logger
        )
        return
    except ClaudeAuthError as exc:
        logger.error("claude 로그인 문제: %s", exc)
        _speak_notice(
            "클로드 코드 로그인이 풀린 것 같아요. 컴퓨터에서 클로드를 실행해 다시 로그인해 주세요.",
            cfg,
            logger,
        )
        return
    except ClaudeTimeout as exc:
        logger.warning("claude 응답 시간 초과: %s", exc)
        _speak_notice("답을 만드는 데 시간이 너무 오래 걸렸어요. 다시 물어봐 주세요.", cfg, logger)
        return
    except ClaudeError as exc:
        logger.error("claude 오류: %s", exc)
        _speak_notice("답을 만들지 못했어요. 잠시 뒤에 다시 물어봐 주세요.", cfg, logger)
        return

    logger.info("답변 완료 (%d자, %.1f초)", len(answer), time.time() - started)
    answer = trim_answer(answer, int(cfg.get("max_answer_chars") or 400))
    speak(answer, cfg, logger)


def run_assistant(cfg: dict, console: bool = False) -> None:
    logger = log
    logger.info("자비스 시작 (v%s, 콘솔=%s)", APP_VERSION, console)

    whisper = WhisperHolder()
    whisper.start(cfg, logger)
    vosk_holder = VoskHolder()

    if bool(cfg.get("announce_start")):
        speak("자비스 준비됐어요.", cfg, logger, cache=True)

    words = ", ".join(f"'{w}'" for w in (cfg.get("wake_words") or []))
    logger.info("부르는 말: %s", words)

    last_error_notice = 0.0
    last_model_notice = 0.0

    while True:
        # 설정을 고쳤을 때 다시 켜지 않아도 반영되도록 매번 다시 읽는다
        cfg.update(load_config())
        try:
            hit = wait_for_wake(cfg, vosk_holder, logger)
        except ModelError as exc:
            logger.error("음성모델 문제: %s", exc)
            now = time.time()
            if now - last_model_notice > 300:
                last_model_notice = now
                _speak_notice(
                    "부르는 말을 알아듣는 준비가 안 됐어요. 설치 파일을 다시 실행해 주세요.",
                    cfg,
                    logger,
                )
            time.sleep(60)
            continue
        except MicError as exc:
            logger.warning("마이크 대기 오류: %s", exc)
            now = time.time()
            if now - last_error_notice > 60:
                last_error_notice = now
                speak("마이크를 찾지 못했어요. 마이크가 꽂혀 있는지 확인해 주세요.", cfg, logger, cache=True)
            time.sleep(5)
            continue

        logger.info("깨우기 말을 들었습니다: '%s' (들은 말: %s)", hit.word, hit.text)
        try:
            _interaction(cfg, whisper, vosk_holder, logger, hit)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.exception("대화 처리 중 오류: %s", exc)
            try:
                speak("잠깐 문제가 생겼어요. 다시 불러 주세요.", cfg, logger, cache=True)
            except Exception:
                pass


def _acquire_single_instance() -> bool:
    """두 번 실행되어 마이크를 서로 잡는 일을 막는다."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 47581))
        sock.listen(1)
    except OSError:
        return False
    globals()["_SINGLETON_SOCKET"] = sock
    return True


# ---------------------------------------------------------------------------
# 시작점
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="자비스 음성 비서 (가입 없이)")
    parser.add_argument("--console", action="store_true", help="로그를 화면에도 보여 주기")
    parser.add_argument("--setup", action="store_true", help="설정 파일 만들기 + 음성모델 내려받기")
    parser.add_argument("--model-path", default="", help="이미 받아 둔 음성모델 폴더 경로")
    parser.add_argument("--check", action="store_true", help="설치 상태 점검")
    parser.add_argument("--test-voice", action="store_true", help="스피커 소리 테스트")
    parser.add_argument("--test-wake", action="store_true", help="부르는 말 인식 테스트")
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

    logger = setup_logging(console=bool(args.console or args.check or args.setup or args.test_voice or args.test_wake))
    logger.info("명령 실행: %s", " ".join(sys.argv[1:]) or "(기본: 실행)")

    cfg = load_config()

    if args.setup:
        return cmd_setup(cfg, args.model_path or None)
    if args.check:
        return cmd_check(cfg)
    if args.test_voice:
        return cmd_test_voice(cfg)
    if args.test_wake:
        return cmd_test_wake(cfg)

    if not _acquire_single_instance():
        logger.info("이미 자비스가 실행 중입니다. 새로 시작하지 않습니다.")
        return 0

    try:
        run_assistant(cfg, console=bool(args.console))
    except KeyboardInterrupt:
        logger.info("사용자가 종료했습니다.")
    except Exception as exc:
        logger.exception("예상하지 못한 오류로 종료합니다: %s", exc)
        return 1
    finally:
        logger.info("자비스 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
