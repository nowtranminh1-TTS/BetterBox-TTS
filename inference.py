"""
Viterbox - Command Line Inference
"""
import argparse
from pathlib import Path

import torch

from viterbox import Viterbox


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Viterbox Text-to-Speech")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--lang", "-l", type=str, default="vi", help="Language (vi/en)")
    parser.add_argument("--ref", "-r", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output file path")
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="Device: cuda / mps / cpu (mặc định: tự chọn cuda → mps → cpu)",
    )

    args = parser.parse_args()
    device = args.device if args.device is not None else _default_device()

    print("Loading model...")
    tts = Viterbox.from_pretrained(device)
    print("✅ Model loaded")
    
    print(f"Generating: '{args.text}'")
    
    audio = tts.generate(
        text=args.text,
        language=args.lang,
        audio_prompt=args.ref,
    )
    
    tts.save_audio(audio, args.output)
    print(f"✅ Saved to: {args.output}")


if __name__ == "__main__":
    main()
