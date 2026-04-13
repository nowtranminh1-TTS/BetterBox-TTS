"""
Viterbox - Vietnamese Text-to-Speech
Based on Chatterbox architecture, fine-tuned for Vietnamese.
"""
import gc
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import math

from pedalboard import Pedalboard as PB, PitchShift
from torch.types import Tensor
from pathlib import Path
from typing import Optional, Union, List

from .models.t3 import T3, T3Config
from .models.t3.modules.cond_enc import T3Cond
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import MTLTokenizer
from .emotional_audio_profiles import get_emotional_audio_profile, get_profile_description

from .tts_helper.tts_support import (
    SEGMENT_TEXT,
    get_reference_sound,
    segment_text,
    punc_norm,
    normalize_text,
    fix_silent_and_speed_audio,
    clearText,
)
from .tts_helper.tts_TTSConds import TTSConds
from .tts_helper.tts_numberToken import getNumberTokenText
from .tts_helper.tts_extension import ViterboxExtensionMixin


class Viterbox(ViterboxExtensionMixin):
    """
    Vietnamese Text-to-Speech model.

    Example:
        >>> tts = Viterbox.from_pretrained("cuda")
        >>> audio = tts.generate("Xin chào!")
        >>> tts.save_audio(audio, "output.wav")
    """

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str = "cuda",
        emotional_profile: Optional[str] = None,
    ):
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.sr = 24000  # Output sample rate
        self.conds: Optional[TTSConds] = None
        # Cache key của audio_prompt lần trước — dùng để tránh re-encode cùng một file
        # Tồn tại trong bộ nhớ khi app đang chạy; reset khi app tắt/khởi động lại
        self._last_audio_prompt_key: Optional[str] = None
        self.emotional_profile = emotional_profile

        # Khởi tạo Pedalboard với profile được chọn
        if not hasattr(self, 'board'):
            if emotional_profile and emotional_profile != "no_eq_processing":
                # Ưu tiên emotional profile nếu có
                self.board = get_emotional_audio_profile(emotional_profile)
                print(f"🎭 Loaded emotional profile: {emotional_profile} - {get_profile_description(emotional_profile)}")
            else:
                self.board = None
                print("🎛️ DONE - Loaded without EQ")

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        emotional_profile: Optional[str] = None,
    ) -> 'Viterbox':
        """Load model từ thư mục modelTTSLocal ở root project."""
        # Thư mục modelTTSLocal nằm ở root project (2 cấp trên viterbox/tts.py)
        local_model_dir = Path(__file__).parent.parent / "modelTTSLocal"

        if not local_model_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục modelTTSLocal tại: {local_model_dir}\n"
                f"Hãy đảm bảo thư mục modelTTSLocal/ tồn tại ở root project."
            )

        return cls.load_local_model(local_model_dir, device, emotional_profile)

    # ── Conditioning ───────────────────────────────────────────────────────────

    def prepare_conditionals(self, audio_prompt: Union[str, Path, torch.Tensor], exaggeration: float = 0.5):
        """
        Prepare conditioning từ reference audio.

        Args:
            audio_prompt: Path to WAV file hoặc audio tensor
            exaggeration: Expression intensity (0.0 - 2.0)
        """
        # Load audio at S3Gen sample rate (24kHz)
        if isinstance(audio_prompt, (str, Path)):
            s3gen_ref_wav, _ = librosa.load(str(audio_prompt), sr=S3GEN_SR, mono=True)
        else:
            s3gen_ref_wav = audio_prompt.cpu().numpy()
            if s3gen_ref_wav.ndim > 1:
                s3gen_ref_wav = s3gen_ref_wav.squeeze()

        # Resample to 16kHz for voice encoder và tokenizer
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        # Limit conditioning length:
        # - speech_pos_emb trained tới 4096 tokens → tối đa ~82s audio mẫu
        # - trừ hao an toàn → giới hạn 80 giây cho cả 3 pipeline bên dưới
        DEC_COND_LEN = S3GEN_SR * 80   # 80s @ 24kHz → cho S3Gen vocoder
        ENC_COND_LEN = S3_SR * 80      # 80s @ 16kHz → cho S3 tokenizer + voice encoder
        s3gen_ref_wav       = s3gen_ref_wav[:DEC_COND_LEN]
        ref_16k_wav_clipped = ref_16k_wav[:ENC_COND_LEN]   # clip 1 lần, dùng chung cho cả 2

        with torch.inference_mode():
            # Get S3Gen conditioning (vocoder)
            s3_cond = self.s3gen.embed_ref(torch.from_numpy(s3gen_ref_wav), S3GEN_SR, device=self.device)

            # Speech cond prompt tokens for T3
            # pipeline: audio đầu vào → S3tokenizer → audio prompt tokens cho T3 conditioning
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward(torch.from_numpy(ref_16k_wav_clipped), max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            # Voice-encoder speaker embedding
            # ref_16k_wav_clipped là phiên bản ĐÃ CẮT NGẮN của ref_16k_wav (tối đa 80s)
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav_clipped], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

            # Create T3Cond
            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        self.conds = TTSConds(t3=t3_cond, s3=s3_cond, ref_wav=torch.from_numpy(s3gen_ref_wav).unsqueeze(0))
        self._last_audio_prompt_key = self._get_audio_prompt_key(audio_prompt)
        return self.conds

    # ── T3 / S3Gen inference helpers ───────────────────────────────────────────

    def _generate_with_T3(
        self, text: str, language: str,
        cfg_weight: float, temperature: float, top_p: float, repetition_penalty: float,
    ) -> Tensor:

        force_fp32: bool = True
        use_autocast = (not force_fp32) and (self.device in ["cuda", "mps"])
        if self.device == "cuda":
            device_type = "cuda"
        elif self.device == "mps":
            device_type = "mps"
        else:
            device_type = None

        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language).to(self.device)

        # CFG: duplicate batch 2 nhánh cond/uncond
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        # Unconditional branch: fill bằng stop_text_token (0) — cũng là <pad> trong CFG
        text_tokens[1].fill_(self.t3.hp.stop_text_token)

        # Thêm token bắt đầu/kết thúc cho text input của T3
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        input_token_count   = text_tokens.shape[-1]
        adaptive_max_tokens = getNumberTokenText(text, input_token_count)

        if use_autocast and device_type is not None:
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16)
        else:
            autocast_ctx = torch.no_grad()

        with autocast_ctx:
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=adaptive_max_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
            )

        speech_tokens = speech_tokens[0]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        return speech_tokens

    def _generate_single(
        self, text: str, language: str,
        cfg_weight: float, temperature: float, top_p: float, repetition_penalty: float,
        speed: float = 1.0,
        pitch_shift: float = 1.0,
    ) -> np.ndarray:
        """Single forward pass → waveform numpy array."""
        with torch.inference_mode():
            speech_tokens = self._generate_with_T3(
                text=text,
                language=language,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Ưu tiên đúng chữ nên giữ nguyên toàn bộ token
            if len(speech_tokens) > 1:
                pass

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.s3,
                speed=speed,
            )

        result = wav[0].cpu().numpy()

        # ── Pitch shift post-processing (Spotify Pedalboard) ──────────────
        # Dùng Pedalboard PitchShift — chất lượng cao hơn librosa rất nhiều
        # pitch_shift: 1.0=giữ nguyên, >1.0=giọng cao, <1.0=giọng trầm
        # Chuyển ratio → semitones: 12 * log2(ratio)
        if pitch_shift != 1.0:
            
            n_semitones = 12.0 * math.log2(max(0.5, min(2.0, float(pitch_shift))))
            try:
                pitch_board = PB([PitchShift(semitones=n_semitones)])
                # Pedalboard cần float32 shape (channels, samples)
                audio_2d = result.reshape(1, -1).astype(np.float32)
                result = pitch_board(audio_2d, self.sr).flatten()
            except Exception as e:
                print(f"⚠️ pitch_shift (pedalboard) failed: {e}")

        del speech_tokens, wav
        gc.collect()
        if torch.cuda.is_available():
            # Dọn bộ nhớ theo lô sau mỗi câu để hạn chế phân mảnh VRAM
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return result

    # ── Main generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        text: str,
        language: str = "vi",
        audio_prompt: Optional[Union[str, Path, torch.Tensor]] = None,
        advance_tts: bool = True,
        skip_processing: bool = False,
        exaggeration: float = 0.0,          # Tăng để âm đuôi mượt hơn, giảm thì âm đuôi cảm xúc mạnh hơn
        cfg_weight: Optional[float] = None,  # None = dùng default 1.0
        temperature: Optional[float] = None, # None = dùng default 0.1
        top_p: Optional[float] = None,       # None = dùng default 0.1
        repetition_penalty: Optional[float] = None,  # None = dùng default 1.0
        speed: float = 1.0,                  # Mel interpolation speed: 0.7~1.5 (1.0=bình thường)
        pitch_shift: float = 1.0,            # F0 scaling pitch: 0.5~2.0 (1.0=bình thường)
    ) -> torch.Tensor:

        cfg_weight         = cfg_weight         if cfg_weight         is not None else 1.0  # tăng: đọc đúng từ hơn
        temperature        = temperature        if temperature        is not None else 0.1  # độ ngẫu nhiên khi chọn token
        top_p              = top_p              if top_p              is not None else 0.1  # giới hạn tập token được chọn
        repetition_penalty = float(repetition_penalty if repetition_penalty is not None else 1.0)  # hơn 1.0 để tránh lặp token
        trailing_silence_ms: int  = 250  # thêm silence đuôi để tránh hai câu sát quá, đọc như đọc rap

        # ── Prepare conditioning ───────────────────────────────────────────────
        # Nếu CÙNG audio_prompt đã dùng trước → tái sử dụng self.conds (cache trong RAM)
        # Nếu KHÁC → encode lại
        # Cache chỉ tồn tại khi app đang mở; tắt/khởi động lại app sẽ reset
        if audio_prompt is not None:
            current_key = self._get_audio_prompt_key(audio_prompt)
            if self.conds is not None and current_key == self._last_audio_prompt_key:
                print("\n♻️♻️♻️ Reusing cached audio conditioning (same audio prompt as before) ♻️♻️♻️")
            else:
                print("\n🆕🆕🆕 Creating new audio prompt...")
                self.prepare_conditionals(audio_prompt, exaggeration)
        elif self.conds is None:
            random_voice = get_reference_sound()
            if random_voice is not None:
                self.prepare_conditionals(random_voice, exaggeration)
            else:
                raise ValueError("No reference audio! Add .wav files to wavs/ folder or provide audio_prompt.")

        # Luôn cập nhật emotion_adv theo giá trị exaggeration hiện tại
        # (không cần re-encode audio — chỉ thay đổi scalar tensor nhẹ)
        # Fix: trước đây khi dùng cached conds, exaggeration cũ bị giữ nguyên
        if self.conds is not None and hasattr(self.conds.t3, 'emotion_adv'):
            self.conds.t3.emotion_adv = exaggeration * torch.ones(1, 1, 1).to(self.device)
            print(f"🤖 🎭 emotion_adv = {exaggeration} | cfg={cfg_weight}, temp={temperature}, top_p={top_p}, rep_pen={repetition_penalty}, speed={speed}, pitch={pitch_shift}\n")

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
                seg_tokens  = self.tokenizer.text_to_tokens(seg["content"], language_id=language)
                token_count = seg_tokens.shape[-1]
                print(f"   [{idx+1}] 🗣  {seg['content']} | Tokens: {token_count}")
                if token_count > 1000:
                    raise ValueError(
                        f"❌ Đoạn [{idx+1}] dài quá {token_count} token (giới hạn 1000), hãy rút ngắn lại."
                    )
            else:
                print(f"   [{idx+1}] ⏸  '{seg['content']}' → {seg['pause_ms']} ms")

        # ── Build audio ────────────────────────────────────────────────────────
        audio_pieces: List[np.ndarray] = []
        join_before:  List[str]        = []
        pending_join: str              = "sentence"

        # Cho phép UI bật/tắt mode theo runtime, không hard-code cố định
        advance_TTS = bool(advance_tts)

# ----------------------------READY FOR INFERENCE TTS--------------------------
        for seg in segments:
            if seg["type"] == SEGMENT_TEXT:

                # giải phóng VRAM trước khi generate để tránh lỗi tích lũy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# --------------------------- INFERENCE TTS with advance mode -----------------------------------
                if advance_TTS:
                    audio_pieces, join_before, pending_join = self.advance_inference_text(
                        seg=seg, language=language, cfg_weight=cfg_weight,
                        temperature=temperature, top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pending_join=pending_join, audio_pieces=audio_pieces,
                        join_before=join_before,
                        speed=speed,
                        pitch_shift=pitch_shift,
                    )
                else:
                    spoken = seg["content"]
                    print(f"\n  🔊 Generating: {spoken}")

                    # chỉ xài trước khi inference, chú ý chỗ xài, không xài trong '_generate_single'
                    getSpoken = punc_norm(spoken, True)

# --------------------------- INFERENCE TTS with normal mode -----------------------------------
                    audio_np = self._generate_single(
                        text=getSpoken,
                        language=language,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        speed=speed,
                        pitch_shift=pitch_shift,
                    )

                    # bỏ đi những khoảng lặng không cần thiết cho TTS NORMAL
                    audio_np = fix_silent_and_speed_audio(audio_np, self.sr,
                                                          threshold_ms=50,
                                                          silence_threshold_db=-60)
                    if len(audio_np) > 0:
                        join_before.append(pending_join)
                        audio_pieces.append(audio_np)
                        pending_join = "sentence"

# ---------------------Dọn sau inference - tránh tích lũy lỗi khi text dài-------------------------
                # Dọn sau mỗi câu để ổn định khi infer nhiều lần liên tiếp
                if torch.cuda.is_available():
                    torch.cuda.synchronize()   # đảm bảo CUDA ops xong hết
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                pending_join = f"pause:{seg['pause_ms']}"

# --------------------------------xử lý hậu kỳ + nối các chuỗi âm thanh rời rạc ----------------------------------------
        if not audio_pieces:
            return torch.zeros(1, self.sr)

        result = audio_pieces[0]
        for i in range(1, len(audio_pieces)):
            rule = join_before[i]
            # Nếu không có pause explicit thì nối liền tự nhiên, không ép silence
            if isinstance(rule, str) and rule.startswith("pause:"):
                ms = int(rule.split(":")[1])   # có dấu câu, cần add khoảng lặng
            else:
                ms = 0                          # không có dấu câu, không cần add khoảng lặng
            silence = np.zeros(int(self.sr * ms / 1000), dtype=result.dtype)
            result  = np.concatenate([result, silence, audio_pieces[i]])

        # nếu dấu câu dài quá, thì giới hạn mức 250ms cho mỗi dấu câu
        result = fix_silent_and_speed_audio(
            result,
            self.sr,
            threshold_ms=250,
            silence_threshold_db=-60,
        )

        # pedalboard hoặc model AI hiện tại trong project, có xử lý được tốc độ nhanh / chậm cho âm thanh?
        # nếu có thì dùng ngay ở đây, không thì bỏ qua

        # Skip audio processing if requested
        if not skip_processing:
            result = self.process_result_audio(result)
        else:
            print("🔇 Skipping output audio processing - using raw audio")

        # BẮT BUỘC CUỐI CÂU PHẢI CÓ KHOẢNG LẶNG NGẮN
        # Trailing silence
        trailing_samples = int(trailing_silence_ms / 1000.0 * self.sr)
        if trailing_samples > 0:
            if result.ndim == 1:
                silence = np.zeros(trailing_samples, dtype=result.dtype)
            else:
                # shape: (samples, channels) hoặc (channels, samples)
                silence = np.zeros((trailing_samples, result.shape[1]), dtype=result.dtype)
            result = np.concatenate([result, silence], axis=0)

        print(f"\n------------------ CREATE TTS ✅ done, CREATE TTS ✅ done, CREATE TTS ✅ done--------------------------")
        print(f"-------------------------------------------------------------------------------------------------------\n")

        return torch.from_numpy(result).unsqueeze(0)

    # ── Advance inference ──────────────────────────────────────────────────────

    def advance_inference_text(
        self, seg: dict, language: str, cfg_weight: float, temperature: float,
        top_p: float, repetition_penalty: float, pending_join: str,
        audio_pieces: List[np.ndarray], join_before: List[str],
        speed: float = 1.0,
        pitch_shift: float = 1.0,
    ) -> tuple:
        spoken = seg["content"]

        getContent   = clearText(spoken.casefold())
        getContent   = normalize_text(getContent, language)
        bunchOfText  = getContent.split()

        print(f"\n  🔊 Generating: {spoken}")

        list_audio_result: List[np.ndarray] = []

        for inferenceAudio in bunchOfText:
            # chỉ xài trước khi inference, chú ý chỗ xài, không xài trong '_generate_single'
            getSpoken = punc_norm(inferenceAudio, True)

            audio_one_word = self._generate_single(
                text=getSpoken,
                language=language,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                speed=speed,
                pitch_shift=pitch_shift,
            )

            if len(audio_one_word) > 0:
                list_audio_result.append(audio_one_word)

        # after inference all words in sentence — combine each item
        if not list_audio_result:
            return audio_pieces, join_before, pending_join

        audio_np = self._stitch_words_for_advance_tts(list_audio_result, 15)

        if len(audio_np) > 0:
            join_before.append(pending_join)
            audio_pieces.append(audio_np)
            pending_join = "sentence"

        return audio_pieces, join_before, pending_join