"""
Minimal OmniVoice wrapper for Gradio app integration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast
import sys

import numpy as np
import torch
try:
    from .omnivoice_support.ttsOmni_Config import inferWithModelOmni
except ImportError:
    from OmniVoice.omnivoice_inference.omnivoice_support.ttsOmni_Config import (
        inferWithModelOmni,
    )

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
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            omni_voice_cls = _import_omnivoice_class()
            model = cast(Any, omni_voice_cls.from_pretrained(
                self.model_path,
                dtype=dtype,
            ))
            model = model.to(self.device)
            self.model = model
        return cast(Any, self.model)

    def load(self):
        return self.loadModelOmni()

    _inferWithModelOmni = inferWithModelOmni

    @property
    def sampling_rate(self) -> int:
        model = cast(Any, self.loadModelOmni())
        return cast(int, model.sampling_rate)


def generate_speech_omni(
    omni: Omni,
    text: str,
    language: str = "vi",
    reference_audio: Optional[str] = None,
    speed: float = 1.0,
):
    if not (text or "").strip():
        return None, "❌ Please enter some text"
    if not reference_audio:
        return None, "❌ No reference audio! Add .wav files to wavs/ folder"

    print("\n🚩bắt đầu inference audio với model OmniVoice\n")
    try:
        audios = omni._inferWithModelOmni(
            text=text.strip(),
            reference_audio=reference_audio,
            language=language,
            speed=speed,
        )
        audio_np = np.asarray(audios[0])
        if audio_np.size == 0:
            print("⚠️ Omni trả về audio rỗng, retry với postprocess_output=False")
            retry_audios = omni._inferWithModelOmni(
                text=text.strip(),
                reference_audio=reference_audio,
                language=language,
                speed=speed,
            )
            audio_np = np.asarray(retry_audios[0])
            if audio_np.size == 0:
                return None, "❌ Omni returned empty audio after retry. Try another reference_audio or shorter text."
        duration = len(audio_np) / omni.sampling_rate
        status = f"✅ Generated (Omni)! | {duration:.2f}s | {language.upper()}"

        print(f"✅ done, đã inference xong với OmniVoice\n")

        return (omni.sampling_rate, audio_np), status
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ Omni error: {str(e)}"