from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    # Relative import: 3 levels up (omnivoice_support -> omnivoice_inference -> OmniVoice root)
    from ...omnivoice.models.omnivoice import OmniVoice
    from ...omnivoice.models.omnivoice import VoiceClonePrompt

# Global cache cho voice_clone_prompt
_voice_clone_cache: dict[str, Any] = {}
_CACHE_FILE = Path(__file__).parent / "voice_clone_prompt_cache.json"


def _get_file_fingerprint(file_path: str) -> str:
    """
    Tạo fingerprint duy nhất cho file bao gồm:
    - File size (để phát hiện thay đổi nhanh)
    - Mtime (thời gian sửa đổi)
    - Hash toàn bộ nội dung file
    """
    try:
        stat = os.stat(file_path)
        file_size = stat.st_size
        mtime = stat.st_mtime

        # Hash toàn bộ file thay vì chỉ 64KB đầu
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Đọc từng chunk để tránh load file lớn vào memory
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    break
                hasher.update(chunk)

        content_hash = hasher.hexdigest()
        # Kết hợp size + mtime + content hash để đảm bảo unique
        return f"{file_size}:{mtime}:{content_hash}"
    except Exception as e:
        print(f"⚠️ Lỗi khi tạo fingerprint: {e}")
        return ""


def _get_cache_key(ref_audio: str, ref_text: Optional[str]) -> str:
    """Tạo unique key từ ref_audio fingerprint + ref_text."""
    # Dùng fingerprint thay vì chỉ path để phát hiện file thay đổi chính xác
    file_fingerprint = _get_file_fingerprint(ref_audio)
    key_data = f"{file_fingerprint}:{ref_text or ''}"
    cache_key = hashlib.md5(key_data.encode('utf-8')).hexdigest()

    # Debug log để trace
    print(f"🔍 Cache key: {cache_key[:8]}... | File: {Path(ref_audio).name} | Size: {file_fingerprint.split(':')[0] if file_fingerprint else '?'}")
    return cache_key


def _load_cache_from_disk() -> dict[str, Any]:
    """Load cache từ file nếu tồn tại."""
    global _voice_clone_cache
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                _voice_clone_cache = json.load(f)
        except Exception:
            _voice_clone_cache = {}
    return _voice_clone_cache


def _save_cache_to_disk():
    """Lưu cache ra file."""
    try:
        with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_voice_clone_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Không thể lưu cache: {e}")


def _get_cached_prompt(cache_key: str) -> Optional[Any]:
    """Lấy cached prompt từ memory hoặc disk."""
    global _voice_clone_cache
    if cache_key in _voice_clone_cache:
        return _voice_clone_cache[cache_key]
    # Thử load từ disk nếu memory cache trống
    _load_cache_from_disk()
    return _voice_clone_cache.get(cache_key)


def _set_cached_prompt(cache_key: str, prompt: Any):
    """Lưu prompt vào cache."""
    global _voice_clone_cache
    _voice_clone_cache[cache_key] = prompt
    _save_cache_to_disk()

def addConfigTextOmni(text: str) -> str:
    # Thêm pause tự nhiên đầu và cuối

    # DO NOT CHANGE THIS CONFIG. WE TEST MANY TIMES, THIS IS THE BEST CONFIG FOR MOST CASES. THANK YOU.
    # change it will cause bugs
    text = " . " + text + " . "         # best config

    return text



# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC
def inferWithModelOmni(
    self,
    text: str,
    reference_audio: str,
    ref_text: Optional[str] = None,  # Thêm transcript của giọng mẫu để clone chính xác hơn
    language: Optional[str] = "vi",
    speed: float = 1.0,
   # duration: Optional[float] = None,  # Thêm để kiểm soát tốc độ đọc cố định
):
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC

    if len(text.strip().split()) == 1:
        text = addConfigTextOmni(text)
        print(f"\n🍂🎧text ĐƠN LẺ trước khi inference TTS {text}\n")
    else:
        text = addConfigTextOmni(text)
        print(f"\n🧩🎧🧩text PHRASE trước khi inference TTS {text}\n")

    print(f" 📑 Reference text cho voice: {ref_text}\n")
    # Chỉnh các tham số cho 'class OmniVoiceGenerationConfig' bên trong thư viện.
    # Chỉ chỉnh sửa ở đây, không động vào bên trong thư viện.

    # num_step: ưu tiên ĐỘ CHÍNH XÁC -> nên tăng vừa phải.
    # Tăng: thường chính xác và mượt hơn (đổi lại chậm hơn). Giảm: nhanh hơn nhưng dễ sai âm/nuốt âm.
    num_step: Optional[int] = 64  # tối thiểu: 8, tối đa: 64 | khuyến nghị chính xác: 48-64

    # guidance_scale: độ bám text/ref.
    # Tăng quá cao: có thể bị "gắt", méo tự nhiên; giảm quá thấp: dễ lệch nội dung. Muốn chính xác: dùng mức trung-cao.
    guidance_scale: Optional[float] = 5.0  # tối thiểu: 0.0, tối đa: 5.0 | khuyến nghị chính xác: 2.5-3.5

    # t_shift: tham số lịch decode.
    # Cho mục tiêu chính xác, giữ gần mặc định để ổn định; tăng/giảm mạnh thường không giúp rõ rệt.
    t_shift: Optional[float] = 0.1  # tối thiểu: 0.05, tối đa: 1.0 | khuyến nghị chính xác: 0.1-0.2

    # layer_penalty_factor: phạt layer sâu để giữ ổn định.
    # Tăng quá cao: có thể mất chi tiết/độ tự nhiên; giảm quá thấp: dễ dao động. Muốn chính xác: mức trung bình.
    layer_penalty_factor: Optional[float] = 10.0  # tối thiểu: 0.0, tối đa: 10.0 | khuyến nghị chính xác: 4.0-6.0

    # position_temperature: THAM SỐ QUAN TRỌNG NHẤT cho tính nhất quán.
    # Tăng: ngẫu nhiên hơn, kết quả mỗi lần khác nhau - Giảm về 0: deterministic, chính xác/lặp lại tốt nhất.
    position_temperature: Optional[float] = 0.0  # tối thiểu: 0.0, tối đa: 8.0 | khuyến nghị chính xác: 0.0

    # class_temperature: quyết định mức "sáng tạo" token.
    # Tăng: dễ sai/lệch phát âm. Giảm về 0: chọn greedy, đúng và ổn định hơn.
    class_temperature: Optional[float] = 0.0  # tối thiểu: 0.0, tối đa: 2.0 | khuyến nghị chính xác: 0.0

    # denoise: nên bật để giảm tạp âm khi clone, thường giúp nghe rõ và chính xác hơn.
    denoise: Optional[bool] = True  # tối thiểu: False, tối đa: True

    # preprocess_prompt: nên bật để làm sạch ref audio trước khi clone, tăng ổn định/độ chính xác.
    preprocess_prompt: Optional[bool] = True  # tối thiểu: False, tối đa: True

    # postprocess_output: ưu tiên chính xác nội dung âm vị -> để False để tránh bị cắt mất âm cuối.
    postprocess_output: Optional[bool] = False  # tối thiểu: False, tối đa: True

    # audio_chunk_duration: mỗi chunk dài hơn thì ít điểm nối hơn (thường ngữ điệu tốt hơn) nhưng tốn VRAM hơn. 
    # giảm thì lại chính xác hơn cho việc đọc text đầu vào
    # nếu text ngắn (< 30-60 giây), nên để audio_chunk_duration = 0 để tắt chunking hoàn toàn.
    audio_chunk_duration: Optional[float] = 0.0  # tối thiểu: 5.0, tối đa: 30.0 | khuyến nghị chính xác: 18-24

    # audio_chunk_threshold: tăng cao để HẠN CHẾ chunking (ít đứt mạch, thường chính xác hơn cho câu vừa/ngắn).
    audio_chunk_threshold: Optional[float] = 60.0  # tối thiểu: 10.0, tối đa: 60.0 | khuyến nghị chính xác: 45-60

    model: OmniVoice = self.loadModelOmni()  # type: ignore[assignment]

    # --- Voice Clone Prompt Caching ---
    # Tạo cache key từ ref_audio và ref_text
    cache_key = _get_cache_key(reference_audio, ref_text)
    cached_prompt_data = _get_cached_prompt(cache_key)

    # Debug chi tiết
    print(f"⚙️ 📂 Đường dẫn đầy đủ: {reference_audio}")
    print(f"⚙️ 🔑 Cache key đầy đủ: {cache_key}")
    print(f"⚙️ 💾 Có trong cache?: {cached_prompt_data is not None}\n")

    voice_clone_prompt = None
    if cached_prompt_data is not None:
        # Reconstruct VoiceClonePrompt từ cache
        try:
            import torch
            from ...omnivoice.models.omnivoice import VoiceClonePrompt
            ref_audio_tokens = torch.tensor(cached_prompt_data['ref_audio_tokens'], dtype=torch.long)
            voice_clone_prompt = VoiceClonePrompt(
                ref_audio_tokens=ref_audio_tokens,
                ref_text=cached_prompt_data['ref_text'],
                ref_rms=cached_prompt_data['ref_rms'],
            )
            print(f"♻️  Sử dụng cached voice_clone_prompt cho: {Path(reference_audio).name}\n")
        except Exception as e:
            print(f"⚠️ Cache corrupted, recreate: {e}")
            voice_clone_prompt = None

    if voice_clone_prompt is None:
        # Tạo mới và cache lại
        print(f"🆕 🔊 ⿻ Tạo voice_clone_prompt mới cho: {Path(reference_audio).name}")
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt if preprocess_prompt is not None else True,
            language=language,
        )
        # Lưu vào cache (convert tensor sang list để JSON serializable)
        cache_data = {
            'ref_audio_tokens': voice_clone_prompt.ref_audio_tokens.cpu().tolist(),
            'ref_text': voice_clone_prompt.ref_text,
            'ref_rms': voice_clone_prompt.ref_rms,
        }
        _set_cached_prompt(cache_key, cache_data)
    # --- End Caching ---

    generate_kwargs = {
        "text": text,
        "language": language,
        # Sử dụng voice_clone_prompt đã cache để tăng tốc batch processing
        "voice_clone_prompt": voice_clone_prompt,
        #"instruct": "female, young adult, high pitch, whisper",
        "speed": speed,
       # "duration": duration,  # Kiểm soát tốc độ đọc cố định (tránh nhanh/ngắn khác nhau)
    }
    if num_step is not None:
        generate_kwargs["num_step"] = num_step
    if guidance_scale is not None:
        generate_kwargs["guidance_scale"] = guidance_scale
    if t_shift is not None:
        generate_kwargs["t_shift"] = t_shift
    if layer_penalty_factor is not None:
        generate_kwargs["layer_penalty_factor"] = layer_penalty_factor
    if position_temperature is not None:
        generate_kwargs["position_temperature"] = position_temperature
    if class_temperature is not None:
        generate_kwargs["class_temperature"] = class_temperature
    if denoise is not None:
        generate_kwargs["denoise"] = denoise
    if preprocess_prompt is not None:
        generate_kwargs["preprocess_prompt"] = preprocess_prompt
    if postprocess_output is not None:
        generate_kwargs["postprocess_output"] = postprocess_output
    if audio_chunk_duration is not None:
        generate_kwargs["audio_chunk_duration"] = audio_chunk_duration
    if audio_chunk_threshold is not None:
        generate_kwargs["audio_chunk_threshold"] = audio_chunk_threshold

    return model.generate(**generate_kwargs)
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC