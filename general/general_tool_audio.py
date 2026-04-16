# general/general_tool_audio.py
from pathlib import Path  # ← ĐÚNG
import re

import numpy as np
from typing import List, Optional
import librosa

# ── Pause config ──────────────────────────────────────────
# Pause durations (ms) per punctuation class
_PUNCT_PAUSE_MS = {
    ".":  450, "!": 450, "?": 450, "。": 450, "！": 450, "？": 450,
    ",":  200, "，": 200, "、": 200,
    ";":  250, "；": 250,
    ":":  200, "：": 200,
    "/":  150, "…": 300, "-": 120, "—": 150, "–": 150,
}

WAVS_DIR = Path("wavs")

# ---------------------------------------------------------------------------
# Segment types used by segment_text()
# ---------------------------------------------------------------------------
SEGMENT_TEXT  = "text"    # A spoken clause → run through T3 + S3
SEGMENT_PAUSE = "pause"   # A punctuation mark → insert silence

def _pause_ms_for(punct: str) -> int:
    return _PUNCT_PAUSE_MS.get(punct, 200)

# ── Audio helpers ─────────────────────────────────────────

def get_reference_sound() -> Optional[Path]:
    """Get reference voice file, fallback to random if not found"""
    if not WAVS_DIR.exists():
        return None
    
    priority_file = WAVS_DIR / "reference_sound.wav"
    if priority_file.exists():
        return priority_file

    voices = list(WAVS_DIR.glob("*.wav"))
    if voices:
        import random
        return random.choice(voices)
    
    return None


def segment_text(text: str) -> List[dict]:
    """
    Convert raw text into an ordered list of typed segment dicts.
    Each dict has the shape:
        {"type": SEGMENT_TEXT,  "content": "<spoken clause>"}
        {"type": SEGMENT_PAUSE, "content": "<punct char>", "pause_ms": <int>}
    Rules
    """
    if not text or not text.strip():
        return []

    punct_pattern = r'(\.{2,}|…+|[.!?,;:/—–\-，。？！、；：])'
    raw_parts = re.split(punct_pattern, text)

    segments: List[dict] = []
    for part in raw_parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if re.fullmatch(punct_pattern, part_stripped):
            punct_key = "…" if re.fullmatch(r'\.{2,}|…+', part_stripped) else part_stripped
            segments.append({
                "type": SEGMENT_PAUSE,
                "content": punct_key,
                "pause_ms": _pause_ms_for(punct_key),
            })
        else:
            segments.append({"type": SEGMENT_TEXT, "content": part_stripped})

    # Merge very short text fragments into previous text segment
    MIN_CHARS = 1
    merged: List[dict] = []
    for seg in segments:
        if (
            seg["type"] == SEGMENT_TEXT
            and len(seg["content"]) < MIN_CHARS
            and merged
            and merged[-1]["type"] == SEGMENT_TEXT
        ):
            merged[-1]["content"] = merged[-1]["content"] + " " + seg["content"]
        else:
            merged.append(seg)

    return merged

def clearText(text: str) -> str:

 # Chuyển toàn bộ về chữ thường
    text = text.casefold()

    original = text

    # Chuẩn hóa mọi dấu câu → ", "
    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\'":\\|,.<>/?`~\.…]+', ', ', text)

    # Dọn khoảng trắng thừa, giữ nguyên từ
    text = " ".join(text.split())
    text = text.strip()
    # Xóa dấu ", " thừa ở đầu và cuối nếu có
    text = text.strip(", ")

    # Fallback nếu sau khi clean thì rỗng
    if not text:
        text = original.strip()

    return text

def addConfigText(text: str) -> str:
    # Thêm pause tự nhiên đầu và cuối

    # DO NOT CHANGE THIS CONFIG. WE TEST MANY TIMES, THIS IS THE BEST CONFIG FOR MOST CASES. THANK YOU.
    # change it will cause bugs
    text = " . " + text + " . "         # best config

    return text


# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ DEPRECATED — Phiên bản cũ, giữ lại để tương thích.
# Phiên bản mới nằm trong tts_precision.py (import ở đầu file).
# Bản mới có: NFC normalize, đọc vocab trực tiếp từ tokenizer JSON,
# lọc invisible chars, phân tích Unicode category, v.v.
# ─────────────────────────────────────────────────────────────────────────────
# BẢNG GỐC RỄ: Giải thích tại sao các trick bên dưới lại hoạt động
# ─────────────────────────────────────────────────────────────────────────────
#
# Pipeline của Viterbox (T3 → speech_tokens → S3Gen):
#
#   [text] → MTLTokenizer.text_to_tokens()
#           → T3.inference() → speech_tokens (acoustic discrete tokens)
#           → S3Gen.inference() → waveform
#
#   T3 là LLaMA-based autoregressive model. Nó nhận text tokens và sinh ra
#   "speech tokens" — một dãy discrete code đại diện cho âm thanh.
#
#   T3 học từ dữ liệu thực, nên nó quen với các pattern "tự nhiên":
#   - Câu đầy đủ, không cụt lủ
#   - Có khoảng lặng (pause) đầu/cuối
#   - Phân cách rõ ràng giữa các từ
#
#   Vì vậy, thêm ký tự `. ` đầu/cuối (như addConfigText) giúp model "tin"
#   rằng đây là một câu hoàn chỉnh, không cần sinh thêm âm đuôi dài.
#
# CÁC TRICK ĐƯỢC IMPLEMENT BÊN DƯỚI:
#   1. Chuẩn hoá khoảng trắng thừa giữa các từ (model nhạy cảm với space)
#   2. Phân tách từ ghép dài bằng cách thêm khoảng trắng nhẹ (long-word splitting)
#   3. Chuẩn hoá dấu hỏi/cảm thán → dấu chấm (tránh model sinh âm lên-xuống bất ngờ)
#   4. Padding pattern " . text . " (đã có ở addConfigText, reuse ở đây)
#   5. Loại bỏ ký tự ngoài bảng vocab tiếng Việt (UNK gây ra âm nhiễu)
#
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str, language: str = "vi") -> str:
    """Normalize text hook (currently pass-through)."""
    return text


def fix_silent_and_speed_audio(
    audio: np.ndarray,
    sr: int, # input framerate
    threshold_ms: int = 50,
    silence_threshold_db: float = -60.0  # ngưỡng dB để xác định silent. càng cao càng là có tiếng nói. VD: -18 dB chắc chắn là người nói
) -> np.ndarray:
    if len(audio) == 0:
        return audio

    # silence_threshold_db: Nếu để quá thấp. VD: -20, có thể bị cắt nhầm vào giọng đọc

    # phần tốc độ đã có model AI xử lý, không cần nữa, vì khi hàm xử lý, dễ bị vỡ âm thanh
    speech_rate = 1.0  # tốc độ nói: 1.0=giữ nguyên, luôn cố định vì speed đã xử lý ở mel-level

    frame_size = int(0.01 * sr)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
    threshold_linear = 10 ** (silence_threshold_db / 20.0)

    is_silent = []
    for frame in frames:
        rms = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0
        is_silent.append(rms < threshold_linear)

    # Gom segments
    segments = []
    current_type = is_silent[0]
    current_start = 0
    for idx in range(1, len(is_silent)):
        if is_silent[idx] != current_type:
            segments.append({
                'silent': current_type,
                'start': current_start * frame_size,
                'end': min(idx * frame_size, len(audio)),
            })
            current_type = is_silent[idx]
            current_start = idx
    segments.append({
        'silent': current_type,
        'start': current_start * frame_size,
        'end': len(audio),
    })

    # Log trước khi xử lý
    print(f"\n  🛠️🛠️🛠️ [before fix_silent_and_speed_audio] threshold={silence_threshold_db}dB | speech_rate={speech_rate}")
    print(f"  {'#':<4} {'type':<8} {'duration_ms':>12} {'rms_db':>10} {'action'}")
    print(f"  {'-'*65}")
    for i, seg in enumerate(segments):
        chunk = audio[seg['start']:seg['end']]
        duration_ms = len(chunk) / sr * 1000
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0
        rms_db = 20 * np.log10(rms + 1e-9)
        seg_type = "silent" if seg['silent'] else "speech"

        if seg['silent']:
            cut = get_cut_silent_ms(duration_ms, threshold_ms)
            if cut > 0:
                action = f"cắt {cut:.0f}ms → còn {max(0, duration_ms - cut):.0f}ms"
            else:
                action = "giữ nguyên (< min)"
        else:
            action = f"speed x{speech_rate}" if speech_rate != 1.0 else "giữ nguyên"

        print(f"  {i+1:<4} {seg_type:<8} {duration_ms:>11.1f}ms {rms_db:>9.1f}dB  {action}")
    print(f"  {'-'*65}\n")

    # Xử lý
    result_parts = []
    for seg in segments:
        chunk = audio[seg['start']:seg['end']]
        if len(chunk) == 0:
            continue

        if seg['silent']:
            get_duration_ms = len(chunk) / sr * 1000

            cut_ms = get_cut_silent_ms(get_duration_ms, threshold_ms)
            
            if cut_ms > 0:
                cut_samples = int(cut_ms / 1000.0 * sr)
                new_len = max(0, len(chunk) - cut_samples)
                chunk = chunk[:new_len]
            result_parts.append(chunk)
        else:
            if speech_rate != 1.0:
                try:
                    chunk = librosa.effects.time_stretch(chunk, rate=speech_rate)
                except Exception as e:
                    print(f"⚠️ time_stretch failed: {e}")
            result_parts.append(chunk)

    if not result_parts:
        return audio

    # Log sau khi xử lý
    total_before = len(audio) / sr * 1000
    total_after = sum(len(p) for p in result_parts) / sr * 1000
    print(f"  ✅⚙️[after fix_silent_and_speed_audio] trước: {total_before:.1f}ms → sau: {total_after:.1f}ms | giảm: {total_before - total_after:.1f}ms\n")

    return np.concatenate(result_parts)

def get_cut_silent_ms(duration_ms: float, threshold_ms: int) -> float:
    if duration_ms <= threshold_ms:
        return 0.0  # không cắt
    else:
        return (duration_ms - threshold_ms) # đảm bảo sau khi cắt, kết quả còn lại luôn là threshold_ms ms
