from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    # Relative import: 3 levels up (omnivoice_support -> omnivoice_inference -> OmniVoice root)
    from ...omnivoice.models.omnivoice import OmniVoice

# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC
def inferWithModelOmni(
    self,
    text: str,
    reference_audio: str,
    language: Optional[str] = "vi",
    speed: float = 1.0,
):
# HẠN CHẾ FIX CHỖ NÀY, VÌ DEV ĐÃ FIX SAO CHO ÂM THANH ĐẦU RA LÀ CHÍNH XÁC NHẤT - ƯU TIÊN ĐỘ CHÍNH XÁC

    # Chỉnh các tham số cho 'class OmniVoiceGenerationConfig' bên trong thư viện.
    # Chỉ chỉnh sửa ở đây, không động vào bên trong thư viện.

    # num_step: ưu tiên ĐỘ CHÍNH XÁC -> nên tăng vừa phải.
    # Tăng: thường chính xác và mượt hơn (đổi lại chậm hơn). Giảm: nhanh hơn nhưng dễ sai âm/nuốt âm.
    num_step: Optional[int] = 64  # tối thiểu: 8, tối đa: 64 | khuyến nghị chính xác: 48-64

    # guidance_scale: độ bám text/ref.
    # Tăng quá cao: có thể bị "gắt", méo tự nhiên; giảm quá thấp: dễ lệch nội dung. Muốn chính xác: dùng mức trung-cao.
    guidance_scale: Optional[float] = 3.0  # tối thiểu: 0.0, tối đa: 5.0 | khuyến nghị chính xác: 2.5-3.5

    # t_shift: tham số lịch decode.
    # Cho mục tiêu chính xác, giữ gần mặc định để ổn định; tăng/giảm mạnh thường không giúp rõ rệt.
    t_shift: Optional[float] = 0.1  # tối thiểu: 0.05, tối đa: 1.0 | khuyến nghị chính xác: 0.1-0.2

    # layer_penalty_factor: phạt layer sâu để giữ ổn định.
    # Tăng quá cao: có thể mất chi tiết/độ tự nhiên; giảm quá thấp: dễ dao động. Muốn chính xác: mức trung bình.
    layer_penalty_factor: Optional[float] = 5.0  # tối thiểu: 0.0, tối đa: 10.0 | khuyến nghị chính xác: 4.0-6.0

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

    # audio_chunk_duration: mỗi chunk dài hơn thì ít điểm nối hơn (thường chính xác ngữ điệu tốt hơn) nhưng tốn VRAM hơn.
    audio_chunk_duration: Optional[float] = 24.0  # tối thiểu: 5.0, tối đa: 30.0 | khuyến nghị chính xác: 18-24

    # audio_chunk_threshold: tăng cao để HẠN CHẾ chunking (ít đứt mạch, thường chính xác hơn cho câu vừa/ngắn).
    audio_chunk_threshold: Optional[float] = 60.0  # tối thiểu: 10.0, tối đa: 60.0 | khuyến nghị chính xác: 45-60

    model: OmniVoice = self.loadModelOmni()  # type: ignore[assignment]
    
    generate_kwargs = {
        "text": text,
        "language": language,
        "ref_audio": reference_audio,
        "speed": speed,
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