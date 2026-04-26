"""
ViterboxExtensionMixin — Các hàm phụ trợ, utility và I/O tách khỏi lớp Viterbox chính.

Viterbox kế thừa mixin này:
    class Viterbox(ViterboxExtensionMixin): ...

Mixin sử dụng self.board, self.sr, self.emotional_profile,
self.conds — được set trong Viterbox.__init__.
"""
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, List
from safetensors.torch import load_file as load_safetensors
from pydub import AudioSegment
from pydub.silence import split_on_silence

from ..models.t3 import T3, T3Config
from ..models.s3gen import S3Gen
from ..models.voice_encoder import VoiceEncoder
from ..models.tokenizers import MTLTokenizer
from general.EQ_emotion_config.eq_emotional_profiles import (
    get_emotional_audio_profile,
    apply_amplitude_envelope,
    list_emotional_profiles as _list_emotional_profiles,
    get_profile_description,
)
from .tts_TTSConds import TTSConds
from .tts_precision import config_token_for_precision
from general.general_tool_audio import clearText, addConfigText


class ViterboxExtensionMixin:
    """
    Mixin chứa: model loader, audio utilities, profile switchers, save_audio.
    Không chứa logic generation chính (generate, _generate_single...).
    """

    # ── Model loader ──────────────────────────────────────────────────────────

    @classmethod
    def load_local_model(
        cls,
        ckpt_dir: Union[str, Path],
        device: str = "cuda",
        emotional_profile: Optional[str] = None,
    ) -> 'ViterboxExtensionMixin':
        """Load model từ thư mục local."""
        ckpt_dir = Path(ckpt_dir)

        # Load Voice Encoder
        ve = VoiceEncoder()
        if device == "cuda":
            ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        else:
            ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location='cpu', weights_only=True))
        ve.to(device).eval()

        # Load T3 model
        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_ml24ls_v2.safetensors")

        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]

        # Resize embeddings nếu vocab size không khớp checkpoint
        if "text_emb.weight" in t3_state:
            old_emb = t3_state["text_emb.weight"]
            if old_emb.shape[0] != t3.hp.text_tokens_dict_size:
                new_emb = torch.zeros((t3.hp.text_tokens_dict_size, old_emb.shape[1]), dtype=old_emb.dtype)
                min_rows = min(old_emb.shape[0], new_emb.shape[0])
                new_emb[:min_rows] = old_emb[:min_rows]
                if new_emb.shape[0] > min_rows:
                    nn.init.normal_(new_emb[min_rows:], mean=0.0, std=0.02)
                t3_state["text_emb.weight"] = new_emb

        if "text_head.weight" in t3_state:
            old_head = t3_state["text_head.weight"]
            if old_head.shape[0] != t3.hp.text_tokens_dict_size:
                new_head = torch.zeros((t3.hp.text_tokens_dict_size, old_head.shape[1]), dtype=old_head.dtype)
                min_rows = min(old_head.shape[0], new_head.shape[0])
                new_head[:min_rows] = old_head[:min_rows]
                if new_head.shape[0] > min_rows:
                    nn.init.normal_(new_head[min_rows:], mean=0.0, std=0.02)
                t3_state["text_head.weight"] = new_head

        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        # Load S3Gen
        s3gen = S3Gen()
        if device == "cuda":
            s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        else:
            s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", map_location='cpu', weights_only=True))
        s3gen.to(device).eval()

        # Load tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "tokenizer_vi_expanded.json"))

        model = cls(t3, s3gen, ve, tokenizer, device, emotional_profile)

        # ── CUDA optimizations (chỉ chạy 1 lần khi load model) ──
        if torch.cuda.is_available():
            # --- TỐI ƯU HÓA CHO ĐỘ CHÍNH XÁC TUYỆT ĐỐI (MAXIMUM ACCURACY) ---
        
            # 1. Dùng FP32 nguyên bản, KHÔNG dùng TF32 xấp xỉ
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # 2. KHÔNG dùng FP16 để cộng dồn (tránh sai số làm tròn cực kỳ quan trọng cho Audio)
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            
            # 3. Ép cuDNN dùng các thuật toán chuẩn xác nhất, loại bỏ sự ngẫu nhiên
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True) # Ép dùng thuật toán Tất Định

            # Attention: CHỈ dùng Math SDP (chính xác tuyệt đối), TẮT Flash Attention và Mem Efficient Attention
            try:
                torch.backends.cuda.enable_math_sdp(True)          # Bật tính toán chính xác truyền thống (Math SDP)
                torch.backends.cuda.enable_flash_sdp(False)        # Tắt FA2 để loại bỏ hoàn toàn sai số xấp xỉ
                torch.backends.cuda.enable_mem_efficient_sdp(False) # Tắt Mem Efficient để đảm bảo bit-exactness
                print("⚡ CUDA optimizations applied (Math SDP for Maximum Accuracy)\n")
            except Exception:
                # PyTorch cũ hơn không có enable_flash_sdp
                print("⚡ CUDA optimizations applied (Maximum Accuracy fallback)\n")
            # channels_last memory layout — Tensor Core friendly cho conv layers (T3, S3Gen)
            import platform
            if platform.system() != "Windows":
                try:
                    t3 = torch.compile(t3, mode="reduce-overhead", dynamic=True)
                    print("⚡ torch.compile applied to T3\n")
                except Exception as e:
                    print(f"⚠️ torch.compile T3 failed: {e}\n")
            else:
                # Ép toàn bộ các model thành phần (T3, S3Gen, VE) sang độ phân giải FP32
                # để ngăn chặn mọi rủi ro tràn số/làm tròn số nếu tệp gốc là FP16
                for _name, _module in [("t3", model.t3), ("s3gen", model.s3gen), ("ve", model.ve)]:
                    setattr(model, _name, _module.to(dtype=torch.float32))
                print("⚡ Windows: Chạy Eager mode FP32 mặc định, ưu tiên ĐỘ CHÍNH XÁC TUYỆT ĐỐI\n")

        # Load default conditioning nếu có sẵn
        conds_path = ckpt_dir / "conds.pt"
        if conds_path.exists():
            model.conds = TTSConds.load(conds_path, device)

        return model

    # ── Cache key ─────────────────────────────────────────────────────────────

    def _get_audio_prompt_key(self, audio_prompt: Union[str, Path, torch.Tensor]) -> str:
        """Tạo cache key từ audio_prompt để so sánh lần sau."""
        if isinstance(audio_prompt, (str, Path)):
            return str(audio_prompt)
        elif isinstance(audio_prompt, torch.Tensor):
            # Dùng data_ptr + shape làm key cho tensor
            return f"tensor:{audio_prompt.data_ptr()}:{audio_prompt.shape}"
        return str(id(audio_prompt))

    # ── Audio processing ──────────────────────────────────────────────────────

    def process_result_audio(self, input_audio: np.ndarray) -> np.ndarray:
        """Xử lý audio với Pedalboard + amplitude envelope tối ưu cho TTS tiếng Việt."""
        # 1. Validate và chuẩn hóa định dạng
        if input_audio.dtype != np.float32:
            input_audio = input_audio.astype(np.float32)

        # 2. Kiểm tra audio không trống
        if len(input_audio) == 0:
            return input_audio

        # 3. Normalization nhẹ — tránh clipping nhưng giữ dynamics
        max_val = np.max(np.abs(input_audio))
        if max_val > 0.95:       # Chỉ normalize nếu quá gần clipping
            input_audio = input_audio * (0.95 / max_val)
        elif max_val < 0.1:      # Boost nếu quá yếu
            input_audio = input_audio * (0.7 / max_val)

        # 4. Xử lý với Pedalboard (EQ + Compression)
        try:
            # Pedalboard cần shape (channels, samples)
            audio_2d = input_audio.reshape(1, -1)
            processed_audio = self.board(audio_2d, self.sr, reset=True).flatten()
        except Exception as e:
            print(f"⚠️ Pedalboard processing error: {e}")
            return input_audio

        # 5. Amplitude envelope — tạo cảm giác emotion qua volume curve
        # "sad" → fade out 25% | "happy" → arch | "dramatic" → fade-in + hold
        if self.emotional_profile:
            processed_audio = apply_amplitude_envelope(
                processed_audio, self.sr, emotion=self.emotional_profile
            )

        # 6. Final safety check
        if np.any(np.isnan(processed_audio)) or np.any(np.isinf(processed_audio)):
            print("⚠️ Invalid audio detected, returning original")
            return input_audio

        return processed_audio

    def _stitch_words_for_advance_tts(
        self, list_audio_result: List[np.ndarray], crossfade_ms: int = 15
    ) -> np.ndarray:
        if not list_audio_result:
            return np.zeros(0, dtype=np.float32)

        combined_segment = None

        for chunk in list_audio_result:
            # đảm bảo mono
            if chunk.ndim > 1:
                chunk = chunk[0]

            # float32 → int16
            samples_int16 = (chunk * 32767.0).astype(np.int16)
            segment = AudioSegment(
                samples_int16.tobytes(),
                frame_rate=self.sr,
                sample_width=2,
                channels=1,
            )

            pieces = split_on_silence(
                segment,
                min_silence_len=20,    # nhỏ để bắt khoảng ngắt giữa chữ
                silence_thresh=-45,    # tùy model
                keep_silence=10,       # giữ padding để không bị cứng
            )

            if not pieces:
                pieces = [segment]

            processed_segment = pieces[0]
            for p in pieces[1:]:
                processed_segment = processed_segment.append(p, crossfade=5)

            if combined_segment is None:
                combined_segment = processed_segment
            else:
                combined_segment = combined_segment.append(processed_segment, crossfade=crossfade_ms)

        final_raw_samples = (
            np.array(combined_segment.get_array_of_samples())
            .astype(np.float32) / 32768.0
        )

        if self.board is not None:
            audio_2d = final_raw_samples.reshape(1, -1)
            processed = self.board(audio_2d, self.sr, reset=True).flatten()
            # Amplitude envelope
            if self.emotional_profile:
                processed = apply_amplitude_envelope(
                    processed, self.sr, emotion=self.emotional_profile
                )
            return processed
        return final_raw_samples

    # ── Profile switchers ─────────────────────────────────────────────────────

    def get_current_profile(self) -> str:
        """Lấy tên profile hiện tại."""
        if self.emotional_profile:
            return f"emotional:{self.emotional_profile}"
        return "no_eq_processing"

    def switch_emotional_profile(self, profile_name: str):
        """Chuyển đổi emotional profile tại runtime."""
        self.board = get_emotional_audio_profile(profile_name)
        self.emotional_profile = profile_name
        print(f"\n🎭 Switched to emotional profile: {profile_name} - {get_profile_description(profile_name)}")

    def list_emotional_profiles(self) -> list:
        """Liệt kê tất cả available emotional profiles."""
        return _list_emotional_profiles()

    def get_emotional_profile_description(self, profile_name: str) -> str:
        """Lấy mô tả của emotional profile."""
        return get_profile_description(profile_name)

    def is_emotional_mode(self) -> bool:
        """Kiểm tra có đang ở emotional mode không."""
        return self.emotional_profile is not None

    # ── Save audio ────────────────────────────────────────────────────────────

    def save_audio(self, audio: torch.Tensor, path: Union[str, Path], trim_silence: bool = False):
        """
        Save audio to file.

        Args:
            audio: Audio tensor from generate()
            path: Output file path
            trim_silence: Whether to trim trailing silence
                          (default: False vì generate() đã trim rồi)
        """
        print(f"\n💾 💿save audio to file: {audio} \n")
        audio_np = audio[0].cpu().numpy()

        if trim_silence:
            audio_np, _ = librosa.effects.trim(audio_np, top_db=30)

        sf.write(str(path), audio_np, self.sr)


# ── Text preprocessing utilities ──────────────────────────────────────────

def punc_norm(text: str, use_precision_config: bool = True) -> str:
    """
    Tiền xử lý text trước khi đưa vào tokenizer.

    Tham số:
        use_precision_config:
            True  (mặc định) = dùng config_token_for_precision từ tts_precision.py
                                (NFC normalize, lọc UNK, tách từ dài, boundary pause)
                                Bản nâng cấp toàn diện — ưu tiên dùng.
            False             = dùng addConfigText cũ (chỉ thêm " . text . ")
    """
    if len(text) == 0:
        return "Bạn cần nhập nội dung để tôi đọc."

    text = clearText(text)

    if use_precision_config:
        # ── Dùng hàm từ tts_precision.py — bao gồm NFC, vocab filter, boundary pause ──
        text = config_token_for_precision(text)
    else:
        # ── Giữ lại cách cũ để tương thích nếu cần ──
        text = addConfigText(text)

    return text
