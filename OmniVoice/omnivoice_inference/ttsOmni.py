"""
Minimal OmniVoice wrapper for Gradio app integration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, cast
import sys

import numpy as np
import torch

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
        self.model = None

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        if model_path:
            return model_path

        local_candidates = [
            Path("modelOmniLocal"),
            Path("OmniVoice/modelOmniLocal"),
        ]
        for candidate in local_candidates:
            if candidate.exists():
                print(f"Model Omni local có tồn tại")
                return str(candidate)

        print(f"Model Omni local KHÔNG tồn tại")
        return "k2-fsa/OmniVoice"

    def loadModelOmni(self):
        if self.model is None:
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            omni_voice_cls = _import_omnivoice_class()
            self.model = omni_voice_cls.from_pretrained(
                self.model_path,
                dtype=dtype,
            )
            self.model = self.model.to(self.device)
        return self.model

    def load(self):
        return self.loadModelOmni()

    def infer(
        self,
        text: str,
        reference_audio: str,
        language: Optional[str] = None,
        speed: float = 1.0,
    ):
        model = self.loadModelOmni()
        print(f"\n🚩bắt đầu inference audio với model OmniVoice\n")
        return model.generate(
            text=text,
            language=language,
            ref_audio=reference_audio,
            speed=speed,
        )

    @property
    def sampling_rate(self) -> int:
        model = self.loadModelOmni()
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

    try:
        audios = omni.infer(
            text=text.strip(),
            reference_audio=reference_audio,
            language=language,
            speed=speed,
        )
        audio_np = np.asarray(audios[0])
        if audio_np.size == 0:
            return None, "❌ Omni returned empty audio. Try another reference_audio or shorter text."
        duration = len(audio_np) / omni.sampling_rate
        status = f"✅ Generated (Omni)! | {duration:.2f}s | {language.upper()}"
        return (omni.sampling_rate, audio_np), status
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ Omni error: {str(e)}"