"""
Model lifecycle helpers for Gradio app.
"""

import gc
import torch

from viterbox import Viterbox
from OmniVoice.omnivoice_inference.ttsOmni import Omni

# Set by app.py during startup
DEVICE = "cpu"

# Lazy singleton model (load on first use)
MODEL = None
OMNI_MODEL = None
ACTIVE_MODEL = None

print("\n\n🎉 Ready for using TTS app 🎉\n\n")


def configure_device(device: str) -> None:
    """Initialize runtime device from main app."""
    global DEVICE
    DEVICE = device

def get_model_viterbox():
    """Load model một lần duy nhất, các lần sau tái sử dụng."""
    global MODEL
    if MODEL is None:
        print("=" * 50)
        print("🚀 Loading Viterbox...")
        print("=" * 50)
        MODEL = Viterbox.from_pretrained(DEVICE)
        print("✅ Model loaded!")
        print("=" * 50)
    return MODEL


def get_omni_model():
    """Load Omni model one time and reuse."""
    global OMNI_MODEL
    if OMNI_MODEL is None:
        print("=" * 50)
        print("🚀 Loading OmniVoice...")
        print("=" * 50)
        OMNI_MODEL = Omni()
        OMNI_MODEL.loadOmniFromUI()
        print("✅ OmniVoice loaded!")
        print("=" * 50)
    return OMNI_MODEL


def _release_viterbox_model():
    global MODEL
    if MODEL is not None:
        MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _release_omni_model():
    global OMNI_MODEL
    if OMNI_MODEL is not None:
        OMNI_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def ensure_active_model(model_choice: str) -> bool:
    """Keep only the selected model in RAM."""
    global ACTIVE_MODEL
    if model_choice == ACTIVE_MODEL:
        print(f"\n👉 🔒[ModelRouter] Keep current model: {model_choice}")
        return False

    if model_choice == "omni":
        print("\n👉 🔄[ModelRouter] Switching: viterbox -> omni")
        _release_viterbox_model()
    else:
        print("\n👉 🔄[ModelRouter] Switching: omni -> viterbox")
        _release_omni_model()

    ACTIVE_MODEL = model_choice
    print(f"\n👉 🚀[ModelRouter] Active model: {ACTIVE_MODEL}")
    return True