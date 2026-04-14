"""
Emotional Audio Profiles v4 — Viterbox TTS
───────────────────────────────────────────
Nguyên tắc:
  1. KHÔNG thay đổi giọng (không EQ, không PitchShift)
  2. Chỉ thay đổi VOLUME theo thời gian (amplitude envelope)
  3. Biên độ thay đổi NHẸ: nhỏ tối đa 10%, to tối đa 15%
  4. Pedalboard chỉ dùng cho safety (limiter) — không chỉnh âm sắc

2 presets:
  sad      → volume giảm nhẹ 10% về cuối câu (1.0 → 0.90)
  question → volume tăng nhẹ 15% về cuối câu (1.0 → 1.15)
"""

import numpy as np
from pedalboard import Pedalboard, HighpassFilter, Limiter
from typing import Dict, Any


class EmotionalAudioProfile:
    """Pedalboard chain tối thiểu — chỉ cleanup + safety, không chỉnh giọng"""

    @staticmethod
    def get_sad_profile() -> Pedalboard:
        """
        Profile BUỒN 💧
        Pedalboard: pass-through (chỉ cleanup + limiter)
        Toàn bộ emotion do amplitude envelope xử lý
        """
        return Pedalboard([
            HighpassFilter(cutoff_frequency_hz=40),   # Chỉ cắt DC/rumble
            Limiter(threshold_db=-1.0),               # Safety
        ])

    @staticmethod
    def get_question_profile() -> Pedalboard:
        """
        Profile CÂU HỎI ❓
        Pedalboard: pass-through (chỉ cleanup + limiter)
        Toàn bộ emotion do amplitude envelope xử lý
        """
        return Pedalboard([
            HighpassFilter(cutoff_frequency_hz=40),   # Chỉ cắt DC/rumble
            Limiter(threshold_db=-0.5),               # Safety (headroom cho phần to)
        ])


# ─── Amplitude envelope ─────────────────────────────────────────────────────
def apply_amplitude_envelope(
    audio: np.ndarray,
    sample_rate: int,
    emotion: str = "neutral",
) -> np.ndarray:
    """
    Thay đổi volume theo thời gian — TẠO CẢM XÚC bằng cao độ âm lượng.

    Quy tắc:
      • Lấy âm hiện tại làm baseline (1.0)
      • Nhỏ tối đa: 0.90  (giảm 10%)
      • To tối đa:  1.15  (tăng 15%)
      • Không EQ, không pitch shift → giọng không bị biến dạng

    "sad"      → 1.0 → 0.90  (nhỏ dần cuối câu — đuối, buồn)
                 Curve: linear nhẹ, tự nhiên
                 Cảm giác: người nói đang mất dần năng lượng

    "question" → 1.0 → 1.15  (to dần cuối câu — lên giọng hỏi)
                 Curve: giữ phẳng 70% đầu, rồi tăng 30% cuối
                 Cảm giác: câu hỏi tiếng Việt, lên giọng ở cuối

    Args:
        audio      : numpy array 1D float32 (waveform)
        sample_rate: Hz (thường 24000)
        emotion    : "sad" | "question" | anything else → no change

    Returns:
        audio đã apply envelope, cùng shape
    """
    n = len(audio)
    if n == 0 or emotion == "neutral":
        return audio

    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    if emotion == "sad":
        # ── NHỎ DẦN VỀ SAU ──────────────────────────────────────────────
        # Linear fade: 1.0 → 0.90 (giảm 10% cuối)
        #
        #   0% câu  → volume 1.00  (bình thường)
        #  25% câu  → volume 0.975 (gần như không khác)
        #  50% câu  → volume 0.95  (hơi nhỏ hơn, tự nhiên)
        #  75% câu  → volume 0.925 (nhỏ dần rõ hơn)
        # 100% câu  → volume 0.90  (nhỏ nhất — vẫn nghe rõ)
        #
        envelope = 1.0 - 0.10 * t

    elif emotion == "question":
        # ── ÂM CUỐI TO HƠN ──────────────────────────────────────────────
        # Tiếng Việt: "Thật không đấy?", "Đúng không á?"
        # → Chữ cuối ("đấy", "á") cần TO ĐỘT NGỘT +40%
        #
        # Thiết kế:
        #   0–75% câu  → volume 1.00  (hoàn toàn bình thường)
        #   75–80% câu → ramp 1.00 → 1.40  (tăng gần tức thì trong ~5%)
        #   80–100% câu → hold 1.40  (giữ to cho chữ cuối)
        #
        #   Ví dụ câu 1 giây (24000 samples):
        #     0–18000   → 1.00  (bình thường)
        #     18000–19200 → ramp lên 1.40  (~50ms, gần đột ngột)
        #     19200–24000 → 1.40  (giữ to)
        #
        ramp_start = int(n * 0.75)    # Bắt đầu tăng tại 75%
        ramp_end   = int(n * 0.80)    # Đạt max tại 80%
        ramp_len   = ramp_end - ramp_start

        envelope = np.ones(n, dtype=np.float32)
        if ramp_len > 1:
            envelope[ramp_start:ramp_end] = np.linspace(1.0, 1.45, ramp_len)
        envelope[ramp_end:] = 1.45    # Hold to cuối câu

    else:
        return audio

    return (audio * envelope).astype(audio.dtype)


# ─── Registry ────────────────────────────────────────────────────────────────
EMOTIONAL_AUDIO_PROFILES: Dict[str, Any] = {
    "sad":      EmotionalAudioProfile.get_sad_profile,
    "question": EmotionalAudioProfile.get_question_profile,
}


def get_emotional_audio_profile(profile_name: str) -> Pedalboard:
    """Lấy Pedalboard đã cấu hình theo tên profile."""
    if profile_name not in EMOTIONAL_AUDIO_PROFILES:
        available = ", ".join(EMOTIONAL_AUDIO_PROFILES.keys())
        raise ValueError(
            f"Emotional profile '{profile_name}' không tồn tại. "
            f"Available: {available}"
        )
    return EMOTIONAL_AUDIO_PROFILES[profile_name]()


def list_emotional_profiles() -> list:
    """Liệt kê tất cả available emotional profiles"""
    return list(EMOTIONAL_AUDIO_PROFILES.keys())


def get_profile_description(profile_name: str) -> str:
    """Lấy mô tả của emotional profile"""
    descriptions = {
        "sad":      "💧 BUỒN — Nhỏ dần 10% cuối câu (1.0→0.90)",
        "question": "❓ CÂU HỎI — To đột ngột +40% ở 15% cuối câu (1.0→1.40)",
    }
    return descriptions.get(profile_name, "Unknown profile")
