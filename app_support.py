"""
app_support.py — Tất cả hàm hỗ trợ và CSS cho Viterbox Gradio UI.
Import toàn bộ file này vào app.py bằng: from app_support import *
"""
import os
import sys
import re
import random
import shutil
import unicodedata
import tempfile
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional

from general.EQ_emotion_config.eq_emotional_profiles import get_profile_description
from viterbox.AI_emotion_config import get_model_emotion_profile


# ── Hằng số ───────────────────────────────────────────────────────────────────
SAVE_FILE = "general/config_path.txt"


def _pyinstaller_bundle_dir() -> Path | None:
    """Thư mục _internal khi chạy PyInstaller onedir/onefile."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return None


def _config_file_path() -> Path:
    """config_path.txt: trong bundle khi frozen, cwd khi dev."""
    bd = _pyinstaller_bundle_dir()
    if bd is not None:
        return bd / SAVE_FILE
    return Path(SAVE_FILE)


# ── Voices ────────────────────────────────────────────────────────────────────

def list_voices() -> list[str]:
    """List available voice files (ưu tiên wavs trong _internal khi chạy exe)."""
    candidates: list[Path] = []
    bd = _pyinstaller_bundle_dir()
    if bd is not None:
        candidates.append(bd / "wavs")
    candidates.append(Path("wavs"))
    for wav_dir in candidates:
        if wav_dir.is_dir():
            return sorted([str(f) for f in wav_dir.glob("*.wav")])
    return []


def _get_default_voice(voices: list[str]) -> str | None:
    """Ưu tiên file mặc định, nếu không có thì lấy file đầu tiên."""
    if not voices:
        return None
    preferred = "reference_sound.wav"
    for v in voices:
        if Path(v).name == preferred:
            return v
    return voices[0]


# ── Config path ───────────────────────────────────────────────────────────────

def save_path(path_text):
    p = _config_file_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(path_text)
    return f"✅ Đã lưu: {path_text}"

def load_path():
    p = _config_file_path()
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return ""  # Trả về trống nếu chưa có file


# ── Save audio ────────────────────────────────────────────────────────────────

def _safe_output_dir(path_text: str) -> Path:
    """Lấy thư mục lưu hợp lệ, fallback về Downloads nếu path không tồn tại."""
    if path_text:
        p = Path(path_text).expanduser()
        if p.exists() and p.is_dir():
            return p
    return Path.home() / "Downloads"


def _slugify_filename_from_text(text: str, max_words: int = 5) -> str:
    """Tạo tên file không dấu, tối đa `max_words` từ."""
    raw = (text or "").strip().lower()
    if not raw:
        return "tts_audio"

    no_accent = unicodedata.normalize("NFD", raw)
    no_accent = "".join(ch for ch in no_accent if unicodedata.category(ch) != "Mn")
    no_accent = re.sub(r"[^a-z0-9\s]", " ", no_accent)
    words = [w for w in no_accent.split() if w][:max_words]
    if not words:
        return "tts_audio"
    return "_".join(words)


def save_generated_audio(audio_data, text: str, folder_path: str, sequence_number):
    """Lưu audio đã sinh vào thư mục chỉ định trong UI."""
    try:
        current_number = int(sequence_number) if sequence_number is not None else 1
    except (TypeError, ValueError):
        current_number = 1
    if current_number < 1:
        current_number = 1

    if audio_data is None:
        return "❌ Chưa có audio để lưu", None, current_number

    sr, audio_np = audio_data
    out_dir = _safe_output_dir(folder_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = _slugify_filename_from_text(text, max_words=5)
    rand_id = random.randint(100, 999)
    prefixed_name = f"{current_number}_{base_name}_{rand_id}.wav"
    out_path = out_dir / prefixed_name
    bd = _pyinstaller_bundle_dir()
    fallback_dir = (bd / "downloads") if bd is not None else Path("downloads")
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fallback_dir / prefixed_name

    # Ưu tiên lưu theo đường dẫn user nhập; lỗi thì fallback downloads (trong _internal nếu chạy exe).
    final_path = out_path
    try:
        sf.write(str(out_path), audio_np, sr)
    except Exception:
        final_path = fallback_path
        try:
            sf.write(str(final_path), audio_np, sr)
        except Exception as e:
            return f"❌ Không thể lưu audio: {str(e)}", None, current_number

    # Trả file trong thư mục temp của app để Gradio không báo InvalidPathError.
    temp_export = Path(os.environ["GRADIO_TEMP_DIR"]) / final_path.name
    try:
        shutil.copyfile(str(final_path), str(temp_export))
    except Exception as e:
        return f"❌ Lưu file xong nhưng không thể xuất file tải xuống: {str(e)}", None, current_number
    return f"✅ Đã lưu audio: {final_path}", str(temp_export), current_number + 1


# ── Generate speech ───────────────────────────────────────────────────────────

def generate_speech_viterbox(
    MODEL,
    text: str,
    language: str = "vi",
    reference_audio=None,
    tts_mode: str = "advance",
    emotional_profile: str = "no_eq_processing",
    ui_exaggeration: float = 1.0,
    model_emotion_profile: str = "AI-precision",
    ai_speed: float = 1.0,
    ui_cfg_weight: float = 1.0,
    ui_temperature: float = 0.1,
    ui_top_p: float = 0.1,
    ui_repetition_penalty: float = 1.0,
    ui_pitch_shift: float = 1.0,
):
    """Generate speech from text"""
    if not text.strip():
        return None, "❌ Please enter some text"

    # Nếu user không upload, dùng dropdown mặc định.
    ref_path = reference_audio

    if ref_path is None:
        return None, "❌ No reference audio! Add .wav files to wavs/ folder"

    try:
        # Handle audio processing options
        if emotional_profile == "no_eq_processing":
            # Skip audio processing entirely
            skip_processing = True
        else:
            # Switch to emotional profile
            MODEL.switch_emotional_profile(emotional_profile)
            skip_processing = False

        # Resolve model emotion profile → override generation parameters
        me_profile = get_model_emotion_profile(model_emotion_profile)
        if me_profile.name != "AI-custom" and me_profile.exaggeration is not None:
            # nếu user chọn cảm xúc từ model AI là khác 'AI-custom' thì KHÔNG SỬ DỤNG bất kỳ tham số nào từ thanh trượt
            effective_exaggeration = me_profile.exaggeration
            gen_cfg_weight = me_profile.cfg_weight
            gen_temperature = me_profile.temperature
            gen_top_p = me_profile.top_p
            gen_repetition_penalty = me_profile.rep_pen
        else:
            effective_exaggeration = ui_exaggeration
            gen_cfg_weight = ui_cfg_weight
            gen_temperature = ui_temperature
            gen_top_p = ui_top_p
            gen_repetition_penalty = ui_repetition_penalty

        # Generate
        wav = MODEL.generate(
            text=text.strip(),
            language=language,
            audio_prompt=ref_path,
            advance_tts=(tts_mode == "advance"),
            skip_processing=skip_processing,
            exaggeration=effective_exaggeration,
            cfg_weight=gen_cfg_weight,
            temperature=gen_temperature,
            top_p=gen_top_p,
            repetition_penalty=gen_repetition_penalty,
            speed=ai_speed,
            pitch_shift=ui_pitch_shift,
        )

        # Convert to numpy
        audio_np = wav[0].cpu().numpy()

        duration = len(audio_np) / MODEL.sr

        # Create status message with audio processing info
        if emotional_profile == "no_eq_processing":
            profile_info = "No EQ Processing"
        else:
            profile_info = get_profile_description(emotional_profile)

        status = f"✅ Generated! | {duration:.2f}s | {language.upper()} | 🎭 {profile_info}"
        if model_emotion_profile != "AI-custom":
            status += f" | 🧠 {me_profile.display_name}"

        return (MODEL.sr, audio_np), status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# ── Voice Profile Builder wrappers ────────────────────────────────────────────

def run_build_voice_profile(MODEL, PRETRAINED_DIR, OUTPUT_DIR, build_voice_profile_fn, exaggeration_val: float) -> str:
    """
    Wrapper để Gradio gọi: build voice profile từ folder viterbox/pretrained/.
    Truyền MODEL đang chạy vào để tái sử dụng — không cần load model mới.

    Log được gom lại thành chuỗi và trả về Textbox trong UI.
    """
    lines = []
    def _log(msg: str):
        print(msg)       # vẫn in ra console để debug
        lines.append(msg)

    build_voice_profile_fn(
        model=MODEL,             # Tái dùng model đang chạy, không load mới
        pretrained_dir=PRETRAINED_DIR,
        output_dir=OUTPUT_DIR,
        exaggeration=exaggeration_val,
        log_fn=_log,
    )
    return "\n".join(lines)


def run_copy_profile_to_model(OUTPUT_DIR, MODEL_DIR, copy_profile_fn) -> str:
    """
    Wrapper để Gradio gọi: copy conds.pt từ viterbox/output-profile/ sang viterbox/modelViterboxLocal/.
    Backup file cũ tự động trước khi ghi đè.
    """
    lines = []
    def _log(msg: str):
        print(msg)
        lines.append(msg)

    result = copy_profile_fn(
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        log_fn=_log,
    )
    lines.append(result)
    return "\n".join(lines)


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0f172a !important; }
.gradio-container { max-width: 100% !important; padding: 1rem 2rem !important; }
.status-badge { 
    display: inline-flex; align-items: center; padding: 4px 12px;
    border-radius: 999px; font-size: 0.8rem; font-weight: 500;
    background: #4f46e5; color: #fff;
}
#main-row { gap: 1rem !important; }
#main-row > div { flex: 1 !important; min-width: 0 !important; }
.card { 
    background: #1e293b !important; border-radius: 0.75rem;
    border: 1px solid #334155 !important; padding: 1rem 1.25rem; height: 100%;
}
.section-title { 
    font-size: 0.85rem; font-weight: 600; color: #e5e7eb;
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;
}
.generate-btn { 
    background: #4f46e5 !important; border-radius: 0.5rem !important;
    font-size: 1rem !important; padding: 10px 24px !important; margin-top: 0.75rem !important;
}
.output-card {
    background: #1e293b !important; border-radius: 0.75rem;
    border: 1px solid #334155 !important; padding: 1rem 1.25rem; margin-top: 0.75rem;
}
/* Accordion Settings: nền trong suốt (hiện nền .card #1e293b phía sau) */
.settings-accordion.block {
    --block-background-fill: transparent !important;
    --block-border-color: transparent !important;
    box-shadow: none !important;
}
"""
