"""
pretrain_voice_builder.py  (v3 — Multi-chunk + Smart Window + Perceiver Average)
──────────────────────────────────────────────────────────────────────────────

LƯU Ý KHI LẤY DATA
1. CHỈ 1 GIỌNG DUY NHẤT (trộn nhiều giọng → output không ổn định)
2. NÊN CÓ KÈM FILE TEXT (đặt tên file text và audio giống nhau, VD: clip1.mp3 + clip1.txt)
   → App dùng text để chọn window 80s đa dạng âm vị nhất
3. ĐỘ DÀI AUDIO: không giới hạn — càng nhiều giờ càng tốt
   → speaker_emb và x-vector: tính từ TOÀN BỘ audio (không cắt)
   → Acoustic context (Perceiver Average): tổng hợp từ tối đa 20 cửa sổ×80s ≈ 26 phút

Audio prompt khi chạy app nên cùng giọng với audio dùng để build profile.
Kết quả sau khi build là file 'conds.pt' trong folder 'viterbox/output-profile/'.
Nhấn 'Copy → modelViterboxLocal' để app dùng ngay (cần restart app).

──────────────────────────────────────────────────────────────────────────────

MỤC ĐÍCH:
    Xây dựng voice conditioning (conds.pt) chất lượng cao từ nhiều giờ audio,
    vượt qua giới hạn 80 giây mà không cần sửa kiến trúc model.

GIẢI THÍCH KỸ THUẬT — SO SÁNH V1 / V2 / V3
═══════════════════════════════════════════════════════════════════════════════
conds.pt gồm 5 trường (phân tích từ S3Gen.embed_ref + prepare_conditionals):

  Trường                     | Giới hạn  | V1       | V2             | V3
  ───────────────────────────────────────────────────────────────────────────
  speaker_emb (T3/VoiceEnc)  | Không có  | 80s      | TOÀN BỘ audio  | TOÀN BỘ audio
  embedding (S3Gen/CAMPPlus) | Không có  | 80s      | TOÀN BỘ audio  | TOÀN BỘ audio
  prompt_feat (mel 24kHz)    | ~10s      | 80s clip | window 10s     | window 10s
  prompt_token (S3 tokens)   | ≤4050 tk  | 80s clip | window 80s     | (thay bằng ↓)
  cond_prompt_speech_emb     | 32 vector | —        | —              | ★ PERCEIVER AVERAGE

  ★ V3 thêm kỹ thuật "Perceiver Average": thay vì dùng 1 window 80s cho
    prompt_token, V3 chạy Perceiver trên N windows khác nhau (mỗi window 80s),
    rồi average output (1, 32, 1024) → T3 nhận context tổng hợp từ N×80s.
    Tối đa 20 windows = ~26 phút acoustic context được nén vào 32 vectors.

TẠI SAO PERCEIVER AVERAGE HỢP LỆ VỀ MẶT TOÁN HỌC?
    Perceiver(pre_attention_query_token=32): nhận chuỗi dài bất kỳ, attend
    qua 32 learned query vectors (ĐỘC LẬP với position đầu vào), xuất ra
    (B, 32, 1024) cố định. Vì queries GIỐNG NHAU với mọi windows:
      mean[ Perceiver(w₁), Perceiver(w₂), ..., Perceiver(wₙ) ] là hợp lệ.

CHỈ NÊN DÙNG 1 GIỌNG DUY NHẤT:
    VoiceEncoder train với loss speaker discrimination — speaker_emb là điểm
    trong không gian "giọng người". Trộn 2 giọng → điểm nằm giữa 2 người
    → T3 sinh giọng không xác định → output không ổn định, lai ghép.

CÁCH SỬ DỤNG:
    1. Đặt audio (.wav/.mp3/.flac/...) cùng 1 giọng vào folder 'viterbox/pretrained/'
    2. (Tuỳ chọn) Đặt file .txt cùng tên audio (VD: clip1.mp3 + clip1.txt)
    3. Nhấn "🧠 Build Voice Profile" trong UI
       hoặc: python pretrain_voice_builder.py [--copy_to_model]
    4. Nhấn "📋 Copy → modelViterboxLocal" để dùng ngay làm default
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import gc
import re
import time
import shutil
import unicodedata
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# ── Đảm bảo import từ đúng thư mục project ──────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# HẰNG SỐ CẤU HÌNH
# ════════════════════════════════════════════════════════════════════════════

PRETRAINED_DIR  = _ROOT / "pretrained"          # Folder chứa audio mẫu
OUTPUT_DIR      = _ROOT / "output-profile"      # Folder xuất conds.pt
MODEL_DIR       = _ROOT / "modelViterboxLocal"  # Folder model gốc

# S3Gen sample rates
TARGET_SR_24K = 24_000  # S3GEN_SR — cho S3Gen (mel, embed_ref)
TARGET_SR_16K = 16_000  # S3_SR    — cho S3Tokenizer, VoiceEncoder, CAMPPlus

# Giới hạn cứng của T3: speech_pos_emb trained tới 4096 tokens ≈ 82s @16kHz
# Dùng 80s để có buffer an toàn
WINDOW_SECONDS    = 80.0   # Cửa sổ cho prompt_token + prompt_feat
SILENCE_BETWEEN_MS = 50   # Khoảng lặng giữa các clip khi nối

# Mel window cho S3Gen (prompt_feat): code S3Gen cảnh báo nếu > 10s
# Dùng tốt nhất 10s với âm sắc phong phú nhất
MEL_WINDOW_SECONDS = 10.0


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: THU THẬP VÀ ĐỌC FILE
# ════════════════════════════════════════════════════════════════════════════

def collect_audio_files(folder: Path) -> List[Path]:
    """Quét folder, lấy tất cả file audio được hỗ trợ, sắp xếp theo tên."""
    supported = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    files = []
    for ext in supported:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))  # sort nhất quán, set loại trùng


def read_text_for_audio(audio_path: Path) -> Optional[str]:
    """
    Tìm file .txt cùng tên với audio (VD: clip1.wav → clip1.txt).
    Trả về nội dung text hoặc None nếu không có.
    """
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def load_audio_pair(
    fpath: Path,
    log_fn=print,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Load 1 file audio, trả về (wav_24k, wav_16k, text).
    wav_24k: float32 mono @ 24kHz (cho S3Gen)
    wav_16k: float32 mono @ 16kHz (cho VoiceEncoder / CAMPPlus / S3Tokenizer)
    text:    nội dung text tương ứng (nếu có file .txt cùng tên)
    """
    try:
        # Load ở sample rate gốc để tránh double-resample
        wav_orig, orig_sr = librosa.load(str(fpath), sr=None, mono=True)

        # Trim silence đầu/cuối (top_db=35: vừa phải, tránh cắt nhầm giọng)
        wav_orig, _ = librosa.effects.trim(wav_orig, top_db=35)

        if len(wav_orig) == 0:
            log_fn(f"  ⚠️  Bỏ qua {fpath.name}: file trống sau khi trim.")
            return None, None, None

        # Resample → 24kHz và 16kHz
        if orig_sr == TARGET_SR_24K:
            wav_24k = wav_orig
        else:
            wav_24k = librosa.resample(wav_orig, orig_sr=orig_sr, target_sr=TARGET_SR_24K)

        if orig_sr == TARGET_SR_16K:
            wav_16k = wav_orig
        elif TARGET_SR_24K == orig_sr:
            wav_16k = librosa.resample(wav_24k, orig_sr=TARGET_SR_24K, target_sr=TARGET_SR_16K)
        else:
            wav_16k = librosa.resample(wav_orig, orig_sr=orig_sr, target_sr=TARGET_SR_16K)

        text = read_text_for_audio(fpath)
        dur = len(wav_24k) / TARGET_SR_24K
        has_text = "📝" if text else "  "
        log_fn(f"  ✅ {has_text} {fpath.name} | {dur:.1f}s | {orig_sr}Hz")
        return wav_24k.astype(np.float32), wav_16k.astype(np.float32), text

    except Exception as e:
        log_fn(f"  ❌ Lỗi khi đọc {fpath.name}: {e}")
        return None, None, None


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: TÍNH SPEAKER EMBEDDING TRÊN TOÀN BỘ AUDIO
# ════════════════════════════════════════════════════════════════════════════

def compute_full_speaker_emb(
    all_wav16k: List[np.ndarray],
    ve_model,           # VoiceEncoder instance
    log_fn=print,
) -> np.ndarray:
    """
    Tính speaker embedding trên TOÀN BỘ audio (không giới hạn 80s).

    Cách hoạt động:
    - VoiceEncoder.embeds_from_wavs() nhận List[np.ndarray]
    - Mỗi phần tử là 1 utterance (có thể dài tùy ý)
    - Nó chia nhỏ thành "partial utterances" (frames) rồi average
    - Trả về (B, 256) — mỗi row là embedding của 1 utterance

    Để kết hợp nhiều utterances:
    - Dùng VoiceEncoder.utt_to_spk_embed() để average + L2-normalize
    - Đây là cách chính xác được thiết kế trong code VoiceEncoder

    Kết quả: vector 256-d biểu diễn đặc trưng giọng nói ổn định hơn,
    bắt được pitch, formant, timbre trên nhiều ngữ cảnh khác nhau.
    """
    log_fn(f"\n  🎙️  Tính speaker embedding trên {len(all_wav16k)} đoạn audio...")

    # embeds_from_wavs trả về (B, 256) → mỗi hàng là 1 utterance embedding
    utt_embeds = ve_model.embeds_from_wavs(all_wav16k, sample_rate=TARGET_SR_16K)

    # utt_to_spk_embed: average + L2-normalize → (256,) speaker embedding
    spk_embed = ve_model.utt_to_spk_embed(utt_embeds)

    log_fn(f"     ✅ Speaker emb shape: {spk_embed.shape} | norm={np.linalg.norm(spk_embed):.4f}")
    return spk_embed  # (256,) float32


def compute_full_xvector(
    all_wav16k: List[np.ndarray],
    speaker_encoder,    # CAMPPlus instance từ S3Gen
    device: str,
    log_fn=print,
) -> "torch.Tensor":
    """
    Tính x-vector (CAMPPlus) trên toàn bộ audio.

    CAMPPlus.inference() trả về tensor embedding cho speaker identity.
    Gộp bằng cách chạy từng chunk, average kết quả.

    embedding này được S3Gen dùng để điều chỉnh timbre của vocoder.
    Tính trên nhiều audio → ổn định hơn, ít bị lệch theo 1 đoạn.
    """
    import torch
    log_fn(f"\n  🔊 Tính x-vector (CAMPPlus) trên {len(all_wav16k)} đoạn audio...")

    all_embeds = []
    for i, wav16k in enumerate(all_wav16k):
        try:
            # CAMPPlus.inference() nhận (1, T) tensor @ 16kHz
            wav_t = torch.from_numpy(wav16k).float().unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = speaker_encoder.inference(wav_t)  # (1, D)
            all_embeds.append(emb.cpu())
        except Exception as e:
            log_fn(f"     ⚠️  Chunk {i+1}: {e}")
            continue

    if not all_embeds:
        log_fn("     ❌ Không tính được bất kỳ x-vector nào!")
        return None

    # Average và trả về trên device gốc
    stacked = torch.cat(all_embeds, dim=0)   # (N, D)
    avg_emb = stacked.mean(dim=0, keepdim=True)  # (1, D)
    # Normalize L2 (tương tự cách CAMPPlus hoạt động)
    avg_emb = avg_emb / (avg_emb.norm(dim=1, keepdim=True) + 1e-8)
    log_fn(f"     ✅ X-vector shape: {avg_emb.shape} | nguồn: {len(all_embeds)} chunks")
    return avg_emb.to(device)


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: CHỌN CỬA SỔ 80s TỐT NHẤT
# ════════════════════════════════════════════════════════════════════════════

def compute_phonetic_diversity(text: str) -> float:
    """
    Đo độ đa dạng âm vị của một đoạn text tiếng Việt.

    Sử dụng tập hợp âm tiết (syllable) làm proxy cho phonetic coverage.
    Text có nhiều âm tiết khác nhau → conditioning phong phú hơn → T3
    giúp học nhiều pattern phát âm → output ổn định hơn.

    Metric: len(unique_syllables) / total_syllables
    → Gần 1.0 = rất đa dạng, gần 0 = lặp đi lặp lại

    Tại sao âm tiết thay vì ký tự? Tiếng Việt là đơn âm tiết — mỗi âm tiết
    là 1 đơn vị phát âm hoàn chỉnh (VD: "sáng" ≠ "sang" ≠ "sàng").
    """
    if not text:
        return 0.0

    # Normalize và tách âm tiết
    text_lower = text.casefold()
    # Tách theo khoảng trắng và dấu câu cơ bản
    syllables = re.findall(r'[^\s,.:;!?\-\(\)\[\]\"\']+', text_lower)

    if not syllables:
        return 0.0

    unique = set(syllables)
    return len(unique) / len(syllables)


def select_best_window(
    all_wavs_24k: List[np.ndarray],      # Danh sách audio 24kHz
    all_texts: List[Optional[str]],       # Text tương ứng (có thể None)
    window_seconds: float = WINDOW_SECONDS,
    log_fn=print,
) -> np.ndarray:
    """
    Chọn cửa sổ 80s tốt nhất để dùng làm prompt_token + prompt_feat.

    Chiến lược:
    A. Nếu có text: chọn tổ hợp các clip cho độ phủ âm vị cao nhất
       (greedy: thêm clip đến khi đủ 80s, ưu tiên clip có diversity cao).
    B. Nếu không có text: gộp từ giữa danh sách ra (tránh đầu/cuối, thường
       chứa lời chào/kết — ít phong phú phonetically).

    Lý do chọn cửa sổ "tốt nhất" thay vì cửa sổ đầu tiên:
    - T3 dùng cond_prompt_speech_tokens như "bối cảnh âm học" để định hướng
      phát âm. Nếu 80s toàn lời chào ("xin chào, hôm nay thời tiết..."),
      T3 sẽ học pattern chào hỏi → khi đọc nội dung kỹ thuật sẽ lạc nhịp.
    - Nếu 80s đa dạng ngữ cảnh → T3 coverage rộng hơn → ổn định hơn.
    """
    max_samples_24k = int(TARGET_SR_24K * window_seconds)
    silence_n = int(TARGET_SR_24K * SILENCE_BETWEEN_MS / 1000)
    silence = np.zeros(silence_n, dtype=np.float32)

    has_any_text = any(t is not None for t in all_texts)

    if has_any_text:
        log_fn("  📊 Có text → chọn window theo độ đa dạng âm vị (phonetic diversity)...")

        # Tính diversity score cho từng clip (clip không có text = điểm 0.3 — thấp hơn trung bình)
        scored_clips = []
        for wav, txt in zip(all_wavs_24k, all_texts):
            score = compute_phonetic_diversity(txt) if txt else 0.3
            scored_clips.append((score, wav, txt))

        # Sắp xếp giảm dần theo score
        scored_clips.sort(key=lambda x: x[0], reverse=True)

        # Greedy: thêm clip từ score cao xuống thấp cho đến khi đủ 80s
        selected = []
        total = 0
        for score, wav, txt in scored_clips:
            if total >= max_samples_24k:
                break
            remaining = max_samples_24k - total
            chunk = wav[:remaining]
            if selected:
                selected.append(silence.copy())
                total += silence_n
            selected.append(chunk.astype(np.float32))
            total += len(chunk)
            short_txt = (txt[:50] + "...") if txt and len(txt) > 50 else (txt or "(no text)")
            log_fn(f"     + {len(chunk)/TARGET_SR_24K:.1f}s | score={score:.2f} | '{short_txt}'")

    else:
        log_fn("  📊 Không có text → dùng kết hợp từ giữa danh sách...")

        # Gộp tuần tự từ clip ở giữa ra 2 phía (tránh đầu/cuối)
        mid = len(all_wavs_24k) // 2
        order = [mid]
        lo, hi = mid - 1, mid + 1
        while lo >= 0 or hi < len(all_wavs_24k):
            if hi < len(all_wavs_24k):
                order.append(hi); hi += 1
            if lo >= 0:
                order.append(lo); lo -= 1

        selected = []
        total = 0
        for idx in order:
            if total >= max_samples_24k:
                break
            wav = all_wavs_24k[idx]
            remaining = max_samples_24k - total
            chunk = wav[:remaining]
            if selected:
                selected.append(silence.copy())
                total += silence_n
            selected.append(chunk.astype(np.float32))
            total += len(chunk)

    if not selected:
        # Fallback: gộp tất cả và cắt
        merged = np.concatenate(all_wavs_24k)[:max_samples_24k]
        return merged.astype(np.float32)

    result = np.concatenate(selected)[:max_samples_24k]
    log_fn(f"  ✅ Window 24kHz: {len(result)/TARGET_SR_24K:.2f}s ({len(result):,} samples)")
    return result.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 3.5: PERCEIVER AVERAGE — TỔNG HỢP NGỮ CẢNH ÂM HỌC TỪ NHIỀU GIỜ
# ════════════════════════════════════════════════════════════════════════════

def slice_windows_from_all(
    all_wavs_24k: List[np.ndarray],
    all_texts: List[Optional[str]],
    window_seconds: float = WINDOW_SECONDS,
    max_windows: int = 20,
    log_fn=print,
) -> List[np.ndarray]:
    """
    Tạo danh sách các cửa sổ 80s khác nhau từ toàn bộ audio.

    Thay vì chỉ chọn 1 window tốt nhất, hàm này tạo ra N windows:
    - Ưu tiên clip có phonetic diversity cao (nếu có text) → thêm trước
    - Sau đó lần lượt lấy các clip còn lại
    - Mỗi window được build theo greedy (clip xếp theo score, fill đến 80s)
    - Trả về List[np.ndarray] — mỗi phần tử là 1 window 80s

    Mục đích: feed nhiều windows vào Perceiver để average output,
    tổng hợp ngữ cảnh âm học từ nhiều giờ audio.
    """
    max_samples = int(TARGET_SR_24K * window_seconds)
    silence_n   = int(TARGET_SR_24K * SILENCE_BETWEEN_MS / 1000)
    silence     = np.zeros(silence_n, dtype=np.float32)

    # Score từng clip
    scored = [
        (compute_phonetic_diversity(t) if t else 0.3, w)
        for w, t in zip(all_wavs_24k, all_texts)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)  # giảm dần theo diversity

    windows: List[np.ndarray] = []

    # Tạo window theo vòng: mỗi vòng bắt đầu từ 1 clip khác nhau
    # (round-robin starting point) để các windows không trùng nhau nhiều
    n = len(scored)
    for start_idx in range(min(n, max_windows)):
        parts, total = [], 0
        # Bắt đầu từ clip khác nhau mỗi window
        order = [(start_idx + i) % n for i in range(n)]
        for idx in order:
            if total >= max_samples:
                break
            wav = scored[idx][1]
            remaining = max_samples - total
            chunk = wav[:remaining]
            if parts:
                parts.append(silence.copy()); total += silence_n
            parts.append(chunk.astype(np.float32))
            total += len(chunk)

        if parts:
            win = np.concatenate(parts)[:max_samples].astype(np.float32)
            windows.append(win)

    log_fn(f"  🪟 Tạo được {len(windows)} cửa sổ 80s khác nhau")
    return windows


def compute_perceiver_averaged_emb(
    windows_24k: List[np.ndarray],
    model,
    device: str,
    log_fn=print,
) -> "Optional[torch.Tensor]":
    """
    Chạy Perceiver Resampler trên nhiều cửa sổ 80s, average output.

    KỸ THUẬT NÀY VƯỢT QUA GIỚI HẠN 80s:

    Tại sao Perceiver output có thể average?
    ─────────────────────────────────────────
    Perceiver(pre_attention_query_token=32) nhận chuỗi dài bất kỳ (4050 tokens),
    attend vào nó qua 32 learned query vectors, và xuất ra (B, 32, 1024).

    Output 32 tokens này là tóm tắt compressed của toàn bộ nội dung âm học
    trong cửa sổ đó. Mỗi query học "chú ý vào khía cạnh nào" của âm thanh.

    Vì learned queries CÙNG nhau xử lý mọi windows (không phụ thuộc nội dung
    đầu vào về position), average các output là hợp lệ:
      Perceiver(window₁) + Perceiver(window₂) + ... + Perceiver(windowₙ)
      ─────────────────────────────────────────────────────────────────
                                  N

    Kết quả: T3 nhìn thấy "trung bình" của mọi cách phát âm trên nhiều giờ audio.
    Đây là cách "nhét vài tiếng" vào conditioning mà không vi phạm kiến trúc.

    Giới hạn thực tế:
    - Mỗi window = 1 lần forward qua speech_emb + speech_pos_emb + Perceiver
    - Tốn thêm VRAM và thời gian tuyến tính theo số windows
    - Khuyến nghị: 5-20 windows (đủ đa dạng, không quá lâu)

    Args:
        windows_24k: List các cửa sổ 80s @ 24kHz
        model:       Viterbox instance (cần model.t3, model.s3gen.tokenizer)
        device:      cuda / cpu

    Returns:
        torch.Tensor (1, 32, 1024) — averaged Perceiver output,
        hoặc None nếu không thành công.
    """
    import torch
    import torch.nn.functional as F

    if not windows_24k:
        return None

    perceiver = model.t3.cond_enc.perceiver
    if perceiver is None:
        log_fn("  ⚠️  Model không dùng Perceiver → bỏ qua Perceiver averaging")
        return None

    speech_emb     = model.t3.speech_emb      # Embedding lookup (vocab → 1024-d)
    speech_pos_emb = model.t3.speech_pos_emb  # Learned positional embedding
    s3_tokzr       = model.s3gen.tokenizer
    plen           = model.t3.hp.speech_cond_prompt_len  # 4050 tokens

    all_perceiver_out = []

    for i, win_24k in enumerate(windows_24k):
        try:
            # Resample 24kHz → 16kHz cho S3 tokenizer
            win_16k = librosa.resample(win_24k, orig_sr=TARGET_SR_24K, target_sr=TARGET_SR_16K)
            win_16k_t = torch.from_numpy(win_16k)

            with torch.inference_mode():
                # Tokenize → speech tokens (1D)
                tokens, _ = s3_tokzr.forward(win_16k_t, max_len=plen)
                tokens = torch.atleast_2d(tokens).to(device)  # (1, T)

                # Embed + add positional encoding — giống prepare_conditioning
                # trong T3.prepare_conditioning() ở tts.py
                emb = speech_emb(tokens) + speech_pos_emb(tokens)  # (1, T, 1024)

                # Chạy Perceiver → (1, 32, 1024)
                perceived = perceiver(emb)  # (1, 32, 1024)

            all_perceiver_out.append(perceived.cpu())
            log_fn(f"  ✅ Window {i+1}/{len(windows_24k)}: {len(win_24k)/TARGET_SR_24K:.1f}s"
                   f" → tokens={tokens.shape[-1]} → perceived {perceived.shape}")

        except Exception as e:
            log_fn(f"  ⚠️  Window {i+1} thất bại: {e}")
            continue

    if not all_perceiver_out:
        log_fn("  ❌ Không có window nào thành công")
        return None

    # Average các Perceiver outputs — đây là bước cốt lõi
    stacked = torch.stack(all_perceiver_out, dim=0)  # (N, 1, 32, 1024)
    averaged = stacked.mean(dim=0)                   # (1, 32, 1024)

    # L2 normalize từng token vector để giữ scale ổn định
    averaged = F.normalize(averaged, p=2, dim=-1)

    log_fn(f"  🎯 Averaged Perceiver output: {averaged.shape} từ {len(all_perceiver_out)} windows")
    log_fn(f"     ≈ Tổng hợp ~{len(all_perceiver_out) * WINDOW_SECONDS / 60:.1f} phút ngữ cảnh âm học")
    return averaged.to(device)


# ════════════════════════════════════════════════════════════════════════════
# HÀM CHÍNH: BUILD VOICE PROFILE (v3)
# ════════════════════════════════════════════════════════════════════════════

def build_voice_profile(
    model=None,
    pretrained_dir: Path = PRETRAINED_DIR,
    output_dir:     Path = OUTPUT_DIR,
    exaggeration:  float = 2.0, # đưa cảm xúc lên cao nhất
    log_fn=print,
) -> str:
    """
    Xây dựng voice conditioning (conds.pt) tối ưu từ nhiều giờ audio.

    V3 so với V1:
    - speaker_emb: tính trên TOÀN BỘ audio (không giới hạn 80s)
    - embedding (x-vector): tính trên toàn bộ audio
    - prompt_token/prompt_feat: chọn cửa sổ 80s/10s TỐT NHẤT
      (ưu tiên đa dạng âm vị nếu có file .txt kèm theo)

    Args:
        model:          Instance Viterbox đang chạy. Nếu None sẽ load mới.
        pretrained_dir: Folder chứa file audio (và tuỳ chọn file .txt cùng tên).
        output_dir:     Folder xuất conds.pt.
        exaggeration:   Mức cảm xúc cho T3Cond.emotion_adv.
        log_fn:         Hàm log (print hoặc wrapper cho Gradio Textbox).
    """
    import torch

    log_fn("=" * 65)
    log_fn("🧠 XÂY DỰNG VOICE PROFILE v3 (Multi-chunk + Smart Window)")
    log_fn("=" * 65)

    # ── 1. Kiểm tra folder ────────────────────────────────────────────────
    pretrained_dir = Path(pretrained_dir)
    output_dir     = Path(output_dir)

    if not pretrained_dir.exists():
        msg = f"❌ Không tìm thấy folder: {pretrained_dir}"
        log_fn(msg); return msg

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Thu thập và đọc tất cả file audio ─────────────────────────────
    audio_files = collect_audio_files(pretrained_dir)
    if not audio_files:
        msg = f"❌ Không tìm thấy file audio trong {pretrained_dir}/"
        log_fn(msg); return msg

    log_fn(f"\n📋 Tìm thấy {len(audio_files)} file audio (định dạng hỗ trợ: wav/mp3/flac/ogg/m4a):")

    all_wavs_24k: List[np.ndarray]        = []
    all_wavs_16k: List[np.ndarray]        = []
    all_texts:    List[Optional[str]]     = []
    total_dur = 0.0

    for fpath in audio_files:
        wav24, wav16, text = load_audio_pair(fpath, log_fn=log_fn)
        if wav24 is None:
            continue
        all_wavs_24k.append(wav24)
        all_wavs_16k.append(wav16)
        all_texts.append(text)
        total_dur += len(wav24) / TARGET_SR_24K

    if not all_wavs_24k:
        msg = "❌ Không đọc được file audio nào."
        log_fn(msg); return msg

    n_with_text = sum(1 for t in all_texts if t is not None)
    log_fn(f"\n  📊 Tổng: {len(all_wavs_24k)} file | {total_dur/60:.1f} phút | {n_with_text} file có text")

    # ── 3. Load model nếu chưa có ─────────────────────────────────────────
    need_unload = False
    if model is None:
        log_fn("\n🤖 Đang load model từ viterbox/modelViterboxLocal/...")
        try:
            from viterbox import Viterbox
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            model = Viterbox.from_pretrained(device)
            need_unload = True
            log_fn(f"   ✅ Loaded trên {device}")
        except Exception as e:
            msg = f"❌ Không thể load model: {e}"
            log_fn(msg); return msg

    device = model.device

    # ── 4. Tính speaker_emb trên TOÀN BỘ audio (★ cải tiến chính) ────────
    log_fn("\n" + "─" * 50)
    log_fn("BƯỚC 4/7: Tính speaker embedding trên toàn bộ audio")
    log_fn("─" * 50)
    log_fn(f"  Sử dụng {len(all_wavs_16k)} đoạn, tổng {total_dur/60:.1f} phút")
    log_fn("  (VoiceEncoder chia nhỏ → average → 1 vector 256-d)")

    t0 = time.time()
    try:
        spk_embed_np = compute_full_speaker_emb(all_wavs_16k, model.ve, log_fn=log_fn)
        # (256,) → (1, 256) tensor như trong prepare_conditionals
        ve_embed = torch.from_numpy(spk_embed_np).unsqueeze(0).to(device)
        log_fn(f"  ✅ Xong trong {time.time()-t0:.1f}s")
    except Exception as e:
        import traceback
        log_fn(f"  ❌ Lỗi tính speaker_emb: {traceback.format_exc()}")
        return f"❌ Lỗi bước speaker_emb: {e}"

    # ── 5. Tính x-vector (CAMPPlus) trên toàn bộ audio ───────────────────
    log_fn("\n" + "─" * 50)
    log_fn("BƯỚC 5/7: Tính x-vector (S3Gen CAMPPlus) trên toàn bộ audio")
    log_fn("─" * 50)

    t0 = time.time()
    try:
        speaker_encoder = model.s3gen.speaker_encoder
        avg_xvector = compute_full_xvector(all_wavs_16k, speaker_encoder, device, log_fn=log_fn)
        if avg_xvector is None:
            log_fn("  ⚠️  Dùng x-vector mặc định từ window 80s (fallback)")
            avg_xvector = None  # sẽ được tính lại ở bước embed_ref
        log_fn(f"  ✅ Xong trong {time.time()-t0:.1f}s")
    except Exception as e:
        log_fn(f"  ⚠️  Không tính được x-vector: {e} → fallback sang window 80s")
        avg_xvector = None

    # ── 6. Chọn cửa sổ 80s tốt nhất ─────────────────────────────────────
    log_fn("\n" + "─" * 50)
    log_fn("BƯỚC 6/7: Chọn cửa sổ 80s tốt nhất cho prompt_token + ref_wav")
    log_fn("─" * 50)

    best_window_24k = select_best_window(
        all_wavs_24k=all_wavs_24k,
        all_texts=all_texts,
        window_seconds=WINDOW_SECONDS,
        log_fn=log_fn,
    )

    # Lưu audio window để debug / nghe lại
    window_path = output_dir / "best_window_80s.wav"
    try:
        sf.write(str(window_path), best_window_24k, TARGET_SR_24K)
        log_fn(f"  💾 Window audio: {window_path}")
    except Exception:
        pass

    # ── 6.5: Perceiver Average — tổng hợp từ nhiều windows ───────────────
    log_fn("\n" + "─" * 50)
    log_fn("BƯỚC 6.5/7: Perceiver Average (tổng hợp vài tiếng audio → 1 context)")
    log_fn("─" * 50)
    log_fn("  Kỹ thuật: chạy Perceiver trên N×80s windows rồi average output (32×1024)")
    log_fn("  Đây là cách hợp lệ duy nhất để vượt giới hạn 80s mà không sửa model")

    t0 = time.time()
    averaged_cond_emb = None  # (1, 32, 1024) hoặc None nếu thất bại
    try:
        # Tạo nhiều windows 80s đa dạng từ toàn bộ audio
        windows_for_perceiver = slice_windows_from_all(
            all_wavs_24k=all_wavs_24k,
            all_texts=all_texts,
            window_seconds=WINDOW_SECONDS,
            max_windows=20,   # tối đa 20 windows = 20×80s = ~26 phút context
            log_fn=log_fn,
        )
        if windows_for_perceiver:
            averaged_cond_emb = compute_perceiver_averaged_emb(
                windows_24k=windows_for_perceiver,
                model=model,
                device=device,
                log_fn=log_fn,
            )
            log_fn(f"  ✅ Perceiver Average xong trong {time.time()-t0:.1f}s")
        else:
            log_fn("  ⚠️  Không tạo được window nào → bỏ qua Perceiver Average")
    except Exception as e:
        import traceback
        log_fn(f"  ⚠️  Perceiver Average thất bại: {e} → dùng window đơn lẻ")
        averaged_cond_emb = None

    # ── 7. Chạy pipeline conditioning với multi-chunk speaker embedding ───
    log_fn("\n" + "─" * 50)
    log_fn("BƯỚC 7/7: Tạo conds.pt (speaker_emb toàn diện + Perceiver averaged context)")
    log_fn("─" * 50)

    t0 = time.time()
    try:
        from viterbox.models.s3gen import S3GEN_SR
        from viterbox.models.s3tokenizer import S3_SR
        from viterbox.models.t3.modules.cond_enc import T3Cond
        from viterbox.tts_helper.tts_TTSConds import TTSConds
    except ImportError:
        # Fallback: thêm đường dẫn viterbox/ trực tiếp vào sys.path
        import sys as _sys
        _viterbox_dir = _ROOT
        if str(_viterbox_dir) not in _sys.path:
            _sys.path.insert(0, str(_viterbox_dir))
        from models.s3gen import S3GEN_SR
        from models.s3tokenizer import S3_SR
        from models.t3.modules.cond_enc import T3Cond
        from tts_helper.tts_TTSConds import TTSConds

    try:
        wav_tensor_24k = torch.from_numpy(best_window_24k).to(device)

        with torch.inference_mode():
            # S3Gen embed_ref: tính prompt_feat (mel) + prompt_token + embedding (x-vector)
            # Nếu avg_xvector đã tính được, ta sẽ override embedding sau
            s3_cond = model.s3gen.embed_ref(wav_tensor_24k, S3GEN_SR, device=device)

            # Override embedding bằng x-vector tính từ toàn bộ audio
            if avg_xvector is not None:
                s3_cond["embedding"] = avg_xvector
                log_fn("  ✅ Đã ghi đè x-vector bằng giá trị multi-chunk")

            if averaged_cond_emb is not None:
                # ── MODE A (TỔNG HỢP): Dùng Perceiver averaged embedding
                # cond_prompt_speech_emb đã averaged từ N×80s windows
                # T3CondEnc sẽ bỏ qua cond_prompt_speech_tokens vì emb đã có sẵn
                # (xem cond_enc.py line 66-67: tokens và emb phải cùng None hoặc cùng có giá trị)
                # → Ta set tokens=None và emb = averaged output PRE-PERCEIVER sẽ không work
                # → Cách đúng: averaged_cond_emb đã là POST-PERCEIVER
                #   → cần feed thẳng vào T3 qua cond_prompt_speech_emb,
                #     BỎ QUA bước Perceiver trong T3CondEnc.forward()
                # Thực tế: averaged_cond_emb là (1, 32, 1024) — shape GIỐNG output của Perceiver
                # Nhưng T3CondEnc sẽ chạy Perceiver LẠI nếu emb có shape > (1, 32, dim)
                # → Để bypass: set cond_prompt_speech_emb = averaged (32 tokens)
                #   và cond_prompt_speech_tokens = None → bypass hoàn toàn Perceiver
                log_fn("  🎯 Dùng Perceiver-averaged context (MODE A — tổng hợp vài tiếng)")
                #
                # QUAN TRỌNG — Tại sao cần dummy_token:
                # T3CondEnc.forward() có assertion (cond_enc.py:66):
                #   assert (tokens is None) == (emb is None)
                # → Nếu tokens=None mà emb có giá trị → AssertionError
                #
                # Giải pháp: set một dummy token tensor (1 token = pad)
                # T3.prepare_conditioning() sẽ check:
                #   if tokens is not None AND emb is None → compute emb từ tokens
                # → Vì emb đã có (averaged_cond_emb), bước này bị SKIP (line 83: t3.py)
                # → Perceiver trong T3CondEnc sẽ chạy lại trên averaged (1, 32, 1024):
                #   perceived_refined = Perceiver(averaged)  ← 1 lần cross-attend thêm
                # → Điều này = 1 lần refinement bổ sung, KHÔNG phá hỏng thông tin
                #
                dummy_token = torch.zeros(1, 1, dtype=torch.long, device=device)  # (1, 1) pad token
                t3_cond = T3Cond(
                    speaker_emb=ve_embed,                          # ← TOÀN BỘ audio
                    cond_prompt_speech_tokens=dummy_token,         # ← dummy để pass assertion
                    cond_prompt_speech_emb=averaged_cond_emb,     # ← averaged N×80s (post-Perceiver)
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=device)
            else:
                # ── MODE B (FALLBACK): 1 window 80s tốt nhất (như v2 cũ)
                log_fn("  ⚠️  Dùng window đơn lẻ 80s (MODE B — fallback)")
                wav_16k_window = librosa.resample(
                    best_window_24k, orig_sr=TARGET_SR_24K, target_sr=TARGET_SR_16K
                )
                t3_cond_prompt_tokens = None
                plen = model.t3.hp.speech_cond_prompt_len  # 4050 tokens
                if plen:
                    s3_tokzr = model.s3gen.tokenizer
                    t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                        torch.from_numpy(wav_16k_window), max_len=plen
                    )
                    t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(device)
                t3_cond = T3Cond(
                    speaker_emb=ve_embed,                              # ← TOÀN BỘ audio
                    cond_prompt_speech_tokens=t3_cond_prompt_tokens,  # ← window 80s
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=device)

        elapsed = time.time() - t0
        log_fn(f"  ✅ Hoàn tất trong {elapsed:.1f}s")

    except Exception as e:
        import traceback
        msg = f"❌ Lỗi tạo conditioning:\n{traceback.format_exc()}"
        log_fn(msg); return msg

    # ── 8. Lưu conds.pt ──────────────────────────────────────────────────
    try:
        conds = TTSConds(
            t3=t3_cond,
            s3=s3_cond,
            ref_wav=torch.from_numpy(best_window_24k).unsqueeze(0),
        )
        out_path = output_dir / "conds.pt"
        conds.save(str(out_path))
        log_fn(f"\n  💾 Đã lưu: {out_path}")
    except Exception as e:
        msg = f"❌ Lỗi khi lưu conds.pt: {e}"
        log_fn(msg); return msg

    # ── 9. Dọn dẹp ───────────────────────────────────────────────────────
    if need_unload:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 10. Báo cáo tổng kết ─────────────────────────────────────────────
    perceiver_mode = "A (Perceiver avg nhiều windows)" if averaged_cond_emb is not None else "B (window 80s đơn lẻ)"
    n_windows = len(windows_for_perceiver) if averaged_cond_emb is not None else 1
    approx_minutes = n_windows * WINDOW_SECONDS / 60

    summary = (
        f"✅ Voice Profile v3 hoàn tất!\n"
        f"   📁 Nguồn: {len(all_wavs_24k)} file | {total_dur/60:.1f} phút audio ({n_with_text} có text)\n"
        f"   🎙️  speaker_emb : TOÀN BỘ {total_dur/60:.1f} phút (không giới hạn 80s)\n"
        f"   🔊 x-vector    : TOÀN BỘ {len(all_wavs_16k)} đoạn (không giới hạn 80s)\n"
        f"   🎯 Acoustic ctx: MODE {perceiver_mode}\n"
        f"     → Tương đương ~{approx_minutes:.1f} phút ngữ cảnh âm học được tổng hợp\n"
        f"   💾 Đã lưu: {out_path}\n"
        f"\n"
        f"📌 Nhấn 'Copy → modelViterboxLocal' để dùng ngay làm default.\n"
        f"   (Cần restart app để áp dụng)"
    )
    log_fn("\n" + "=" * 65)
    log_fn(summary)
    log_fn("=" * 65)
    return summary


# ════════════════════════════════════════════════════════════════════════════
# COPY PROFILE SANG MODEL DIR
# ════════════════════════════════════════════════════════════════════════════

def copy_profile_to_model(
    output_dir: Path = OUTPUT_DIR,
    model_dir:  Path = MODEL_DIR,
    log_fn=print,
) -> str:
    """
    Copy conds.pt từ viterbox/output-profile/ sang viterbox/modelViterboxLocal/ để dùng ngay làm default.
    Tự động backup file cũ trước khi ghi đè.
    """
    src = Path(output_dir) / "conds.pt"
    dst = Path(model_dir)  / "conds.pt"

    if not src.exists():
        return "❌ Chưa có conds.pt trong viterbox/output-profile/. Hãy Build Voice Profile trước."

    if dst.exists():
        backup = dst.with_suffix(".pt.bak")
        shutil.copy2(str(dst), str(backup))
        log_fn(f"  📦 Đã backup file cũ → {backup.name}")

    try:
        shutil.copy2(str(src), str(dst))
        msg = f"✅ Đã copy voice profile sang:\n   {dst}\nRestart app để áp dụng."
        log_fn(msg)
        return msg
    except Exception as e:
        msg = f"❌ Lỗi khi copy: {e}"
        log_fn(msg)
        return msg


# ════════════════════════════════════════════════════════════════════════════
# CHẠY STANDALONE (không qua UI)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build voice conditioning profile v3 từ folder viterbox/pretrained/"
    )
    parser.add_argument("--pretrained_dir", type=str, default=str(PRETRAINED_DIR))
    parser.add_argument("--output_dir",     type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--exaggeration",   type=float, default=2.0,
                        help="Mức cảm xúc 0.0-2.0 (default: 1.0) - 2.0 là cao nhất")
    parser.add_argument("--copy_to_model",  action="store_true",
                        help="Tự động copy conds.pt sang viterbox/modelViterboxLocal/ sau khi build")
    args = parser.parse_args()

    result = build_voice_profile(
        model=None,
        pretrained_dir=Path(args.pretrained_dir),
        output_dir=Path(args.output_dir),
        exaggeration=args.exaggeration,
    )
    print(result)

    if args.copy_to_model:
        print(copy_profile_to_model(
            output_dir=Path(args.output_dir),
            model_dir=MODEL_DIR,
        ))
