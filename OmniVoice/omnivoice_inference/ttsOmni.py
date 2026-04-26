"""
Minimal OmniVoice wrapper for Gradio app integration.
"""
from __future__ import annotations

# Set HF Hub env vars BEFORE importing transformers to disable warnings
import os
import math
from pedalboard import Pedalboard as PB, PitchShift

# Disable telemetry: Prevent Hugging Face from sending usage statistics/analytics
# Điều này tránh các request ngầm đến HF Hub để báo cáo dữ liệu sử dụng
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Force offline mode: Không gọi API đến HF Hub, chỉ dùng model local
# Điều này tránh warning "unauthenticated requests" vì không còn request nào được gửi đi
os.environ["HF_HUB_OFFLINE"] = "1"

# Set dummy token để tránh warning "unauthenticated requests"
# Vì đang ở offline mode, token này sẽ không được sử dụng cho bất kỳ request nào
# nhưng sẽ làm hài lòng auth check của huggingface_hub
os.environ["HF_TOKEN"] = "dummy"

# Disable symlink warning: Tránh warning về việc Windows không hỗ trợ symlinks tốt
# (thường xuất hiện khi HF Hub cố tạo symlink cho cache files)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathlib import Path
from typing import Any, Optional, cast
import sys
import gc
import warnings
import tempfile, os
import numpy as np
import torch

try:
    from .omnivoice_support.ttsOmni_Config import inferWithModelOmni
except ImportError:
    from OmniVoice.omnivoice_inference.omnivoice_support.ttsOmni_Config import (  # type: ignore
        inferWithModelOmni,
    )

# Thêm thư mục gốc của project vào sys.path để import được module 'general'
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from general.general_tool_audio import (  # type: ignore
    SEGMENT_TEXT,
    get_reference_sound,
    segment_text,
    normalize_text,
    fix_silent_and_speed_audio,
    clearText,
    create_srt_file
)
from general.noise_detect_VAD import vad_trim  # type: ignore

def _import_omnivoice_class():
    try:
        from omnivoice.models.omnivoice import OmniVoice as OmniVoiceClass
        return OmniVoiceClass
    except ModuleNotFoundError:
        # Fallback when OmniVoice is present as local source (repo checkout).
        local_omnivoice_root = Path(__file__).resolve().parents[1]
        local_omnivoice_root_str = str(local_omnivoice_root)
        if local_omnivoice_root_str not in sys.path:
            sys.path.insert(0, local_omnivoice_root_str)
        from omnivoice.models.omnivoice import OmniVoice as OmniVoiceClass
        return OmniVoiceClass


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Omni:
    """Lazy-loaded OmniVoice model wrapper."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or _best_device()
        self.model_path = self._resolve_model_path(model_path)
        self.model: Optional[Any] = None

    @staticmethod
    def _validate_local_model_dir(model_dir: Path) -> None:
        required_files = [
            model_dir / "config.json",
        ]
        weight_candidates = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin.index.json",
        ]

        missing = [p for p in required_files if not p.exists()]
        has_any_weight = any(p.exists() for p in weight_candidates)
        if not has_any_weight:
            # Show the expected weight file names for clarity.
            missing.extend(weight_candidates)

        if missing:
            print("❌ Thiếu file quan trọng trong model Omni local. Không thể load:")
            for p in missing:
                print(f"- {p.as_posix()}")
            raise FileNotFoundError(
                f"Omni local model incomplete at '{model_dir.as_posix()}'. Missing required files."
            )

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        if model_path:
            return model_path

        candidate = Path("OmniVoice/modelOmniLocal")
        if candidate.exists():
            if not candidate.is_dir():
                print("❌ Model Omni local path tồn tại nhưng không phải thư mục:")
                print(f"- {candidate.as_posix()}")
                raise NotADirectoryError(candidate.as_posix())

            Omni._validate_local_model_dir(candidate)
            print("🏠 Model Omni local có tồn tại\n")
            return str(candidate)

        print("Model Omni local KHÔNG tồn tại")
        return "k2-fsa/OmniVoice"

    def loadModelOmni(self) -> Any:
        if self.model is None:
            dtype = torch.float32 # Ép toàn bộ về FP32 cho độ chính xác cao nhất
            omni_voice_cls = _import_omnivoice_class()
            model = cast(Any, omni_voice_cls.from_pretrained(
                self.model_path,
                dtype=dtype,
            ))
            model = model.to(self.device)

            self.model = model
            print("📝 ASR chỉ load khi cần xài\n")
        return cast(Any, self.model)

    def loadOmniFromUI(self):
        return self.loadModelOmni()

    _inferWithModelOmni = inferWithModelOmni

    @property
    def sampling_rate(self) -> int:
        if not hasattr(self, '_sampling_rate'): # chưa có trong cache thì load model để lấy sampling_rate
            model = cast(Any, self.loadModelOmni())
            self._sampling_rate = cast(int, model.sampling_rate) # đã cache
        return self._sampling_rate


def generate_speech_omni(
    omni: Omni,
    text: str,
    language: str = "vi",
    reference_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    speed: float = 1.0,
    pitch_shift: float = 1.0,            # F0 scaling pitch: 0.5~2.0 (1.0=bình thường)
):
    if not (text or "").strip():
        return None, "❌ Please enter some text", None
    if not reference_audio:
        ref_path = get_reference_sound()
        if ref_path is None:
            return None, "❌ No reference audio! Add .wav files to wavs/ folder", None
        reference_audio = str(ref_path)

    # ── Preprocess text ────────────────────────────────────────────────────
    text = clearText(text)
    text = normalize_text(text, language)

    # Segment — tách câu theo dấu câu
    segments = segment_text(text)
    if not segments:
        segments = [{"type": SEGMENT_TEXT, "content": text}]

    # Log segments
    text_segs = [s for s in segments if s["type"] == SEGMENT_TEXT]
    print(f"📝 Text segmented into {len(segments)} items ({len(text_segs)} spoken chunks):")

    for idx, seg in enumerate(segments):
        if seg["type"] == SEGMENT_TEXT:
            print(f"   [{idx+1}] 🗣  [{seg['content']}]")
        else:
            print(f"   [{idx+1}] ⏸  '{seg['content']}' → {seg['pause_ms']} ms")

    # ── Build audio ────────────────────────────────────────────────────────
    audio_pieces: list[np.ndarray] = []
    join_before:  list[str]        = []
    pending_join: str              = "sentence"

    # ── Create SRT file ─────────────────────────────────────────────────────
    arrSrt: list[dict]      = []  # để làm file SRT, List chứa {startTime, endTime, text}
    current_time: float      = 0.0  # Thời gian tích lũy (giây)

# ----------------------------READY FOR INFERENCE TTS--------------------------
    print(f"\n🚩bắt đầu inference audio với model OmniVoice: speed={speed}, pitch={pitch_shift}", flush=True)

    # Debug: Show speed effect on estimated duration
    if speed != 1.0:
        print(f"   📊 Speed {speed} sẽ tạo audio {'ngắn hơn' if speed > 1 else 'dài hơn'} ~{abs(speed-1)*100:.0f}% so với speed=1.0", flush=True)
    for seg in segments:
        if seg["type"] == SEGMENT_TEXT:

            spoken = seg["content"]

            print(f"\n===================================================")
            print(f"\n  🔊📢🔊 Omni Generating: {spoken}\n", flush=True)

            audios = omni._inferWithModelOmni(
                text=spoken.strip(),
                reference_audio=reference_audio,
                ref_text=ref_text,
                language=language,
                speed=speed,
            )

            # audios là list, lấy phần tử đầu tiên
            getFirstAudio = audios[0]

            # Debug: kiểm tra âm thanh đầu ra
            if hasattr(getFirstAudio, 'shape'):
                max_amp = float(np.max(np.abs(getFirstAudio))) if len(getFirstAudio) > 0 else 0.0
                print(f"   🐛 [DEBUG] getFirstAudio: shape={getFirstAudio.shape}, max_amp={max_amp:.4f}")
            else:
                print(f"   🐛 [DEBUG] getFirstAudio: type={type(getFirstAudio)}")

            # giữ lại speech, bỏ non-speech
            getFirstAudio = vad_trim(getFirstAudio, omni.sampling_rate, margin_s=0.05)
            
            audio_np = fix_silent_and_speed_audio(getFirstAudio, omni.sampling_rate,
                                                  threshold_ms=50,
                                                  silence_threshold_db=-45)

            # ── Pitch shift post-processing (Spotify Pedalboard) ──────────────
            # Dùng Pedalboard PitchShift — chất lượng cao hơn librosa rất nhiều
            # pitch_shift: 1.0=giữ nguyên, >1.0=giọng cao, <1.0=giọng trầm
            # Chuyển ratio → semitones: 12 * log2(ratio)
            if pitch_shift != 1.0:
                
                n_semitones = 12.0 * math.log2(max(0.5, min(2.0, float(pitch_shift))))
                try:
                    pitch_board = PB([PitchShift(semitones=n_semitones)])
                    # Pedalboard cần float32 shape (channels, samples)
                    audio_2d = audio_np.reshape(1, -1).astype(np.float32)
                    audio_np = pitch_board(audio_2d, omni.sampling_rate).flatten()
                except Exception as e:
                    print(f"⚠️ pitch_shift (pedalboard) failed: {e}")                                      

            # [SRT FILE] Tạo timing item cho segment này
            segment_duration = len(audio_np) / omni.sampling_rate
            start_time = current_time
            end_time = current_time + segment_duration
            
            timing_item = {
                "startTime": start_time,
                "endTime": end_time,
                "text": spoken
            }
            arrSrt.append(timing_item)
            
            # [SRT FILE]Cập nhật current_time cho segment tiếp theo
            current_time = end_time
            
            print(f"  🎵 Audio generated: {len(audio_np)} samples | {start_time:.3f}s - {end_time:.3f}s", flush=True)
            if len(audio_np) > 0:
                join_before.append(pending_join)
                audio_pieces.append(audio_np)
                pending_join = "sentence"

            # ---------------------đảm bảo CUDA ops xong hết-------------------------
            if torch.cuda.is_available():
                torch.cuda.synchronize()   # đảm bảo CUDA ops xong hết
            # Không gọi empty_cache() ở đây để tránh phân mảnh VRAM và giảm tốc độ
        else:
            # [SRT FILE] Cộng thời gian pause của dấu câu vào current_time
            pause_seconds = seg['pause_ms'] / 1000.0
            current_time = current_time + pause_seconds

            pending_join = f"pause:{seg['pause_ms']}"

 # --------------------------------xử lý hậu kỳ + nối các chuỗi âm thanh rời rạc ----------------------------------------
    if not audio_pieces:
        return None, "❌ Không tạo được âm thanh từ text", None

    print(f"\n🔢 🧩 count audio_pieces: {len(audio_pieces)}, count join_before: {len(join_before)}")
    for idx, piece in enumerate(audio_pieces):
        print(f"   audio_pieces[{idx}]: shape={piece.shape}, dtype={piece.dtype}")
    print(f"   join_before: {join_before}")

    #-------- nối các chuỗi âm thanh rời rạc
    result = audio_pieces[0].astype(np.float32)
    for i in range(1, len(audio_pieces)):
        rule = join_before[i]
        # Nếu không có pause explicit thì nối liền tự nhiên, không ép silence
        if isinstance(rule, str) and rule.startswith("pause:"):
            ms = int(rule.split(":")[1])   # có dấu câu, cần add khoảng lặng
        else:
            ms = 0                          # không có dấu câu, không cần add khoảng lặng
        silence = np.zeros(int(omni.sampling_rate * ms / 1000), dtype=np.float32)
        piece = audio_pieces[i].astype(np.float32)
        result  = np.concatenate([result, silence, piece])
        print(f"🔗Concatenated piece {i}: result len={len(result)}")

    # KHÔNG SỬ DỤNG 'fix_silent_and_speed_audio'. 
    # vì user nhập dấu câu thế nào thì khoảng lặng giữa các câu giữ nguyên như config
   

    # BẮT BUỘC CUỐI CÂU PHẢI CÓ KHOẢNG LẶNG NGẮN
    trailing_silence_ms: int  = 250  # thêm silence đuôi để tránh hai câu sát quá, đọc như đọc rap

    trailing_samples = int(trailing_silence_ms / 1000.0 * omni.sampling_rate)
    if trailing_samples > 0:
        if result.ndim == 1:
            silence = np.zeros(trailing_samples, dtype=result.dtype)
        else:
            # shape: (samples, channels) hoặc (channels, samples)
            silence = np.zeros((trailing_samples, result.shape[1]), dtype=result.dtype)
        result = np.concatenate([result, silence], axis=0)

    duration = len(result) / omni.sampling_rate
    status = f"✅ Generated (Omni)! | {duration:.2f}s | {language.upper()}"

    # [SRT FILE] Tạo file SRT trong temp directory để Gradio có thể trả về
    
    gradio_temp = os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir())
    srt_temp_path = os.path.join(gradio_temp, f"omni_srt_{hash(text) % 1000000}.srt")
    create_srt_file(arrSrt, srt_temp_path)

    print(f"✅ Created SRT file: {srt_temp_path}", flush=True)

    print(f"\n✅ done, đã inference xong với OmniVoice và tạo file SRT | duration={duration:.2f}s\n", flush=True)
    print(f"===========================================================================================================")
    print(f"===========================================================================================================\n\n\n")

    # Dọn dẹp VRAM một lần duy nhất sau khi hoàn thành toàn bộ text
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return (omni.sampling_rate, result), status, srt_temp_path