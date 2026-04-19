import torch
import numpy as np
from pathlib import Path
import librosa
import sys
import importlib.util

# Add current directory to path for importing local module
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Load Silero VAD model from local directory
model_dir = _current_dir / "silero_vad_model_local"

# Global VAD model
_VAD_MODEL = None
_VAD_UTILS = None


def get_vad_model():
    """Load Silero VAD model from local path (singleton)"""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        try:
            # Import from local package (now fixed with relative imports)
            from silero_vad_model_local import (
                get_speech_timestamps, read_audio,
                save_audio, VADIterator, collect_chunks, drop_chunks
            )
            from silero_vad_model_local.utils_vad import init_jit_model

            # Load model from local jit file
            model_path = model_dir / "data" / "silero_vad.jit"
            model = init_jit_model(str(model_path))

            # Create utils tuple (same format as torch.hub.load returns)
            _VAD_UTILS = (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
                drop_chunks
            )
            _VAD_MODEL = model
        except Exception as e:
            print(f"⚠️ Could not load Silero VAD from local: {e}")
            return None, None
    return _VAD_MODEL, _VAD_UTILS


def vad_trim(audio: np.ndarray, sr: int, margin_s: float = 0.01) -> np.ndarray:
    """
    Aggressive trim using Silero VAD - only keep clean speech, remove ALL artifacts.
    
    Args:
        audio: Audio array (numpy)
        sr: Sample rate
        margin_s: Margin to keep around speech (seconds)
    """
    if len(audio) == 0:
        return audio
        
    model, utils = get_vad_model()
    if model is None:
        return trim_silence(audio, sr, top_db=20)
        
    (get_speech_timestamps, _, _, _, collect_chunks, _) = utils
    
    try:
        # Always resample to 16k for VAD (most accurate)
        vad_sr = 16000
        if sr != vad_sr:
            wav_16k = librosa.resample(audio, orig_sr=sr, target_sr=vad_sr)
        else:
            wav_16k = audio
        wav_tensor = torch.tensor(wav_16k, dtype=torch.float32)
        
        # AGGRESSIVE VAD settings - only keep clear speech
        timestamps = get_speech_timestamps(
            wav_tensor, 
            model, 
            sampling_rate=vad_sr, 
            threshold=0.5,           # HIGHER threshold = Chỉ giữ speech rõ ràng, bỏ breath/noise mờ
            neg_threshold=0.35,       # Lower neg threshold = Thoát khỏi speech nhanh hơn khi có dấu hiệu silence
            min_speech_duration_ms=50, # Bỏ các đoạn rè rè ngắn < 50ms
            min_silence_duration_ms=50, # Phát hiện silence nhanh hơn
            speech_pad_ms=30,          # Padding nhỏ, không giữ nhiễu thừa
            max_speech_duration_s=10.0, # Chỉ giữ các đoạn speech, cắt bỏ hoàn toàn silence đầu/cuối/giữa
        )
        
        if not timestamps:
            # No clear speech detected - return minimal audio or silence
            return np.zeros(0, dtype=audio.dtype)
        
        # Collect only speech segments, concatenate them (remove ALL silence between)
        speech_audio = collect_chunks(timestamps, wav_tensor)
        
        # Convert back to numpy
        speech_np = speech_audio.numpy()
        
        # Resample back to original sr if needed
        if sr != vad_sr:
            speech_np = librosa.resample(speech_np, orig_sr=vad_sr, target_sr=sr)
        
        return speech_np
        
    except Exception as e:
        print(f"⚠️ VAD Error: {e}")
        return trim_silence(audio, sr, top_db=20)

def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Legacy trim silence (energy based)."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed