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

# giảm 50ms mỗi dấu câu
# _PUNCT_PAUSE_MS = {
#     ".":  400, "!": 400, "?": 400, "。": 400, "！": 400, "？": 400,
#     ",":  150, "，": 150, "、": 150,
#     ";":  200, "；": 200,
#     ":":  150, "：": 150,
#     "/":  100, "…": 250, "-": 70, "—": 100, "–": 100,
# }

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

def clearText(text: str, language: str = "en") -> str:

 # Chuyển toàn bộ về chữ thường
    text = text.casefold()

    if language.lower() == "vi" :
        text = convertNumuberToVietNamText(text)

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
# trick 4 áp dụng cho các model kiểu cũ: text -> audio token -> audio wave
# CÁC TRICK ĐƯỢC IMPLEMENT BÊN DƯỚI:
#   1. Chuẩn hoá khoảng trắng thừa giữa các từ (model nhạy cảm với space)
#   2. Phân tách từ ghép dài bằng cách thêm khoảng trắng nhẹ (long-word splitting)
#   3. Chuẩn hoá dấu hỏi/cảm thán → dấu chấm (tránh model sinh âm lên-xuống bất ngờ)
#   4. Padding pattern " . text . " (đã có ở addConfigText, reuse ở đây)
#   5. Loại bỏ ký tự ngoài bảng vocab tiếng Việt (UNK gây ra âm nhiễu)
#
# ─────────────────────────────────────────────────────────────────────────────

def convertNumuberToVietNamText(text: str) -> str:
    """
    trong hàm này, sẽ convert những con số sang tiếng việt.
    Ví dụ về quy luật như sau:  
    số 1 -> thì thành 'một'
    số 11 -> thì thành 'mười một'
    số 111 -> thì thành 'một trăm mười một'
    số 1111 -> thì thành 'một ngàn một trăm mười một'

    nếu câu sau: 'hôm nay là thứ 6, ngày 12 tháng 11 năm 1234' -> thì câu sau khi chuyển là: 'hôm nay là thứ sáu, ngày mười hai tháng mười một năm một ngàn hai trăm ba mươi tư' 
    """
    def num_to_vietnamese(n: int) -> str:
        if n == 0:
            return "không"
            
        digit_names = {
            0: "không",
            1: "một",
            2: "hai",
            3: "ba",
            4: "bốn",
            5: "năm",
            6: "sáu",
            7: "bảy",
            8: "tám",
            9: "chín"
        }
        
        # unit names when tens digit is 1
        unit_names_ten_1 = {
            0: "",
            1: "một",
            2: "hai",
            3: "ba",
            4: "bốn",
            5: "lăm",
            6: "sáu",
            7: "bảy",
            8: "tám",
            9: "chín"
        }
        
        # unit names when tens digit is > 1
        unit_names_ten_gt1 = {
            0: "",
            1: "mốt",
            2: "hai",
            3: "ba",
            4: "tư",
            5: "lăm",
            6: "sáu",
            7: "bảy",
            8: "tám",
            9: "chín"
        }
        
        group_units = ["", "ngàn", "triệu", "tỷ", "ngàn tỷ", "triệu tỷ", "tỷ tỷ"]
        
        n_str = str(n)
        
        groups = []
        while n_str:
            groups.append(n_str[-3:])
            n_str = n_str[:-3]
            
        group_words = []
        for i, group in enumerate(groups):
            digits = [int(d) for d in group]
            is_leading = (i == len(groups) - 1)
            
            if sum(digits) == 0:
                continue
                
            words = []
            if is_leading:
                if len(digits) == 1:
                    u = digits[0]
                    words.append(digit_names[u])
                elif len(digits) == 2:
                    t, u = digits
                    if t == 1:
                        words.append("mười")
                        if u != 0:
                            words.append(unit_names_ten_1[u])
                    else:
                        words.append(digit_names[t])
                        words.append("mươi")
                        if u != 0:
                            words.append(unit_names_ten_gt1[u])
                elif len(digits) == 3:
                    h, t, u = digits
                    words.append(digit_names[h])
                    words.append("trăm")
                    if t == 0 and u == 0:
                        pass
                    elif t == 0 and u > 0:
                        words.append("lẻ")
                        words.append(digit_names[u])
                    elif t == 1:
                        words.append("mười")
                        if u != 0:
                            words.append(unit_names_ten_1[u])
                    else:
                        words.append(digit_names[t])
                        words.append("mươi")
                        if u != 0:
                            words.append(unit_names_ten_gt1[u])
            else:
                h, t, u = digits
                words.append(digit_names[h])
                words.append("trăm")
                if t == 0 and u == 0:
                    pass
                elif t == 0 and u > 0:
                    words.append("lẻ")
                    words.append(digit_names[u])
                elif t == 1:
                    words.append("mười")
                    if u != 0:
                        words.append(unit_names_ten_1[u])
                else:
                    words.append(digit_names[t])
                    words.append("mươi")
                    if u != 0:
                        words.append(unit_names_ten_gt1[u])
                        
            if group_units[i]:
                words.append(group_units[i])
                
            group_words.append(" ".join(w for w in words if w))
            
        group_words.reverse()
        return " ".join(group_words)

    def replace_match(m):
        num_str = m.group(0)
        viet_text = num_to_vietnamese(int(num_str))
        print(f"  🔢 Số: {num_str} -> Chữ: {viet_text}")
        return viet_text

    return re.sub(r'\d+', replace_match, text)


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


# tool tạo SRT file
def _format_srt_time(seconds: float) -> str:
    """Format time as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_file(timing_items: List[dict], output_path: str) -> str:
    """
    Create SRT subtitle file from timing items.
    
    Args:
        timing_items: List of dict with keys: startTime, endTime, text
        output_path: Path to save the SRT file
    
    Returns:
        Path to the created SRT file
    """
    output_path = Path(output_path)
    
    srt_lines = []
    for idx, item in enumerate(timing_items, start=1):
        start = _format_srt_time(item["startTime"])
        end = _format_srt_time(item["endTime"])
        text = item["text"]
        
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(f"{text}")
        srt_lines.append("")  # Empty line between entries
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write SRT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    
    return str(output_path)
