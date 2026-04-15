import traceback
from general.EQ_emotion_config.eq_emotional_profiles import get_profile_description
from viterbox.AI_emotion_config import get_model_emotion_profile

def generate_speech_viterbox(
    MODEL,
    text: str,
    language: str = "vi",
    reference_audio=None,
    tts_mode: str = "advance",
    emotional_profile: str = "no_eq_processing",
    ui_exaggeration: float = 1.0,
    model_emotion_profile: str = "AI-precision",
    ai_speed: float = 1.0,
    ui_cfg_weight: float = 1.0,
    ui_temperature: float = 0.1,
    ui_top_p: float = 0.1,
    ui_repetition_penalty: float = 1.0,
    ui_pitch_shift: float = 1.0,
):
    """Generate speech from text"""
    if not text.strip():
        return None, "❌ Please enter some text"

    # Nếu user không upload, dùng dropdown mặc định.
    ref_path = reference_audio

    if ref_path is None:
        return None, "❌ No reference audio! Add .wav files to wavs/ folder"

    try:
        # Handle audio processing options
        if emotional_profile == "no_eq_processing":
            # Skip audio processing entirely
            skip_processing = True
        else:
            # Switch to emotional profile
            MODEL.switch_emotional_profile(emotional_profile)
            skip_processing = False

        # Resolve model emotion profile → override generation parameters
        me_profile = get_model_emotion_profile(model_emotion_profile)
        if me_profile.name != "AI-custom" and me_profile.exaggeration is not None:
            # nếu user chọn cảm xúc từ model AI là khác 'AI-custom' thì KHÔNG SỬ DỤNG bất kỳ tham số nào từ thanh trượt
            effective_exaggeration = me_profile.exaggeration
            gen_cfg_weight = me_profile.cfg_weight
            gen_temperature = me_profile.temperature
            gen_top_p = me_profile.top_p
            gen_repetition_penalty = me_profile.rep_pen
        else:
            effective_exaggeration = ui_exaggeration
            gen_cfg_weight = ui_cfg_weight
            gen_temperature = ui_temperature
            gen_top_p = ui_top_p
            gen_repetition_penalty = ui_repetition_penalty

        # Generate
        wav = MODEL.generate(
            text=text.strip(),
            language=language,
            audio_prompt=ref_path,
            advance_tts=(tts_mode == "advance"),
            skip_processing=skip_processing,
            exaggeration=effective_exaggeration,
            cfg_weight=gen_cfg_weight,
            temperature=gen_temperature,
            top_p=gen_top_p,
            repetition_penalty=gen_repetition_penalty,
            speed=ai_speed,
            pitch_shift=ui_pitch_shift,
        )

        # Convert to numpy
        audio_np = wav[0].cpu().numpy()

        duration = len(audio_np) / MODEL.sr

        # Create status message with audio processing info
        if emotional_profile == "no_eq_processing":
            profile_info = "No EQ Processing"
        else:
            profile_info = get_profile_description(emotional_profile)

        status = f"✅ Generated! | {duration:.2f}s | {language.upper()} | 🎭 {profile_info}"
        if model_emotion_profile != "AI-custom":
            status += f" | 🧠 {me_profile.display_name}"

        return (MODEL.sr, audio_np), status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"
