"""
Viterbox - Gradio Web Interface
"""
import torch
import warnings
import gradio as gr

warnings.filterwarnings('ignore')

import tempfile, os

os.environ["GRADIO_TEMP_DIR"] = tempfile.gettempdir() + "/my_gradio_tmp"
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

from pathlib import Path
from viterbox import Viterbox
from viterbox.emotional_audio_profiles import list_emotional_profiles, get_profile_description
from viterbox.AI_emotion_config import get_model_emotion_choices
from pretrain_voice_builder import build_voice_profile, copy_profile_to_model, PRETRAINED_DIR, OUTPUT_DIR, MODEL_DIR
from app_support import (
    CSS,
    list_voices, _get_default_voice,
    save_path, load_path,
    save_generated_audio,
    generate_speech,
    run_build_voice_profile, run_copy_profile_to_model,
)

# Load model
print("=" * 50)
print("🚀 Loading Viterbox...")
print("=" * 50)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

MODEL = Viterbox.from_pretrained(DEVICE)
print("✅ Model loaded!")
print("=" * 50)

print("\n\n🎉 Ready for using TTS app 🎉\n\n")


# ── Wrapper functions (inject MODEL + dirs into app_support functions) ─────────

def _generate_speech(text, language, reference_audio, tts_mode,
                     emotional_profile, exaggeration, model_emotion_profile,
                     ai_speed, ui_pitch_shift, ui_cfg_weight, ui_temperature, ui_top_p, ui_repetition_penalty):
    return generate_speech(
        MODEL, text, language, reference_audio, tts_mode,
        emotional_profile, exaggeration, model_emotion_profile,
        ai_speed, ui_cfg_weight, ui_temperature, ui_top_p, ui_repetition_penalty, ui_pitch_shift,
    )

def _run_build_voice_profile(exaggeration_val):
    return run_build_voice_profile(MODEL, PRETRAINED_DIR, OUTPUT_DIR, build_voice_profile, exaggeration_val)

def _run_copy_profile_to_model():
    return run_copy_profile_to_model(OUTPUT_DIR, MODEL_DIR, copy_profile_to_model)

def _update_model_emotion_controls(profile):
    is_custom = profile == "AI-custom"
    return (
        gr.update(interactive=is_custom),
        gr.update(interactive=is_custom),
        gr.update(interactive=is_custom),
        gr.update(interactive=is_custom),
    )


# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🎙️ Viterbox TTS",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate"),
    css=CSS,
    js="""
    function() {
        setTimeout(() => {
            document.querySelectorAll('textarea').forEach(el => {
                el.setAttribute('spellcheck', 'false');
            });
        }, 1000);
    }
    """
) as demo:

    gr.HTML("""
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <h1 style="margin: 0; color: #6b7280; font-size: 2rem;">🎙️ Betterbox TTS</h1>
            <p style="color: #6b7280; margin-top: 0.5rem;">Based on app Viterbox TTS</p>
        </div>
    """)

    gr.HTML('<div style="text-align: center; margin-bottom: 1rem;"><span class="status-badge">🎯 Fine-tuned Model</span></div>')

    with gr.Row(equal_height=True, elem_id="main-row"):
        # Left - Text Input
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="section-title">📝 Text Input</div>')

            language = gr.Radio(
                choices=[("🇻🇳 Tiếng Việt", "vi"), ("🇺🇸 English", "en")],
                value="vi", label="Language"
            )

            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Nhập văn bản cần đọc...",
                lines=5
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")

            # ── Voice Profile Builder ─────────────────────────────────────
            with gr.Accordion("🧠 Voice Profile Builder", open=False):
                with gr.Column(elem_classes=["card"]):
                    gr.HTML(
                        '<div style="font-size:0.9rem; color:#ffffff; margin-bottom:0.5rem;">'
                        'Gộp audio trong pretrained/ → tạo conditioning tối ưu → lưu vào output-profile/. '
                        'Nhấn Copy để dùng ngay làm default (cần restart app).'
                        '</div>'
                    )
                    with gr.Row():
                        build_profile_btn = gr.Button(
                            "🧠 Build Voice Profile",
                            variant="primary",
                            size="sm",
                            scale=3,
                        )
                        copy_profile_btn = gr.Button(
                            "📋 Copy → modelTTSLocal",
                            variant="secondary",
                            size="sm",
                            scale=2,
                        )
                    build_profile_output = gr.Textbox(
                        label="Build Log",
                        lines=6,
                        interactive=False,
                        placeholder="Nhấn 'Build Voice Profile' để bắt đầu...",
                    )

            # nhập thứ tự audio để save
            with gr.Row():
                numeric_input = gr.Number(
                    label="Thứ tự:",
                    value=1,
                    minimum=1,
                    step=1,
                    precision=0,
                    interactive=True,
                )

        # Right - Voice & Settings
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="section-title">🎤 Reference Voice</div>')

            wav_files = list_voices()
            default_voice = _get_default_voice(wav_files)
            if wav_files:
                ref_dropdown = gr.Dropdown(
                    choices=[(Path(f).stem, f) for f in wav_files],
                    label="Select Voice",
                    value=default_voice,
                )
            else:
                ref_dropdown = gr.Dropdown(choices=[], label="No voices in wavs/")

            ref_audio = gr.Audio(
                label="Or Upload/Record",
                type="filepath",
                value=default_voice,
                sources=["upload", "microphone"],
            )

            # ---Setting -----------------------------------------------
            with gr.Accordion("⚙️ Settings", open=False):
                with gr.Column(elem_classes=["card"]):

                    tts_mode = gr.Radio(
                        choices=[("TTS normal", "normal"), ("TTS advance", "advance")],
                        value="normal",
                        label="TTS Mode",
                        info="Normal: theo câu | Advance: theo từng từ",
                    )

                    # Emotional Audio Selection
                    with gr.Row():
                        emotional_choices = [
                            ("no_eq_processing")
                        ]
                        for profile in list_emotional_profiles():
                            description = get_profile_description(profile)
                            emotional_choices.append((description, profile))

                        emotional_profile = gr.Dropdown(
                            choices=emotional_choices,
                            value="no_eq_processing",
                            label="🎭 Emotional Audio - EQ for output audio",
                            info="Chọn cảm xúc cho giọng nói (No Processing = audio gốc, không qua xử lý)",
                        )

                    with gr.Row():
                        ui_pitch_shift = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            step=0.05,
                            value=1.0,
                            label="🎵 Pitch Shift - for output audio",
                            info="Cao độ giọng nói. 1.0=bình thường, >1=giọng cao, <1=giọng trầm. Không đổi tốc độ.",
                        )

                    with gr.Row():
                        ai_speed = gr.Slider(
                            minimum=0.7,
                            maximum=1.5,
                            step=0.05,
                            value=1.0,
                            label="🏎️ AI Speed (Mel Interpolation) - for AI input",
                            info="Tốc độ giọng nói từ model AI. 1.0=bình thường, >1=nhanh, <1=chậm. Giữ nguyên pitch.",
                        )

                    with gr.Row():
                        model_emotion = gr.Dropdown(
                            choices=get_model_emotion_choices(),
                            value="AI-precision",
                            label="🧠 Model AI Emotion Profile - for AI input",
                            info="Profile cảm xúc từ tham số model AI (override exaggeration + cfg + temp + top_p khi chọn)",
                        )

                    # ── Advanced AI Parameters ─────────────────────────────────
                    with gr.Accordion("🧪 Advanced AI Parameters - for AI input", open=False):
                        gr.HTML(
                            '<div style="font-size:0.82rem; color:#94a3b8; margin-bottom:0.5rem;">'
                            'Tham số điều khiển trực tiếp model AI. Khi chọn Model Emotion Profile khác Default, '
                            'các slider CFG/Temp/Top-P sẽ bị override bởi profile đó.'
                            '</div>'
                        )

                        exaggeration = gr.Slider(0, 2, 2, step=0.1, label="exaggeration - emotion - for AI input",
                                                info="cảm xúc. 0: âm đuôi cảm xúc mạnh, 2: âm đuôi mượt hơn",
                                                interactive=True)

                        with gr.Row():
                            ui_cfg_weight = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                                value=2.0,
                                label="📏 CFG Weight",
                                info="Cao=đọc đúng từ, Thấp=tự do+chậm hơn",
                            )
                            ui_temperature = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                step=0.01,
                                value=0.1,
                                label="🌡️ Temperature",
                                info="Cao=prosody đa dạng, Thấp=ổn định",
                            )

                        with gr.Row():
                            ui_top_p = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                step=0.01,
                                value=0.1,
                                label="🎯 Top-P",
                                info="Cao=token đa dạng, Thấp=an toàn",
                            )

                            # phạt lặp từ (repetition_penalty). thấp là tuân thủ chính xác, cao thì có thể bỏ từ lặp
                            ui_repetition_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                step=0.05,
                                value=1.0,
                                label="🔁 Repetition Penalty",
                                info=">1.0 tránh lặp token, quá cao sẽ cứng",
                            )

    # Save download folder
    with gr.Row():
        # value=load_path() giúp tự động hiện lại nội dung cũ khi mở App
        folder_input = gr.Textbox(
            label="Download Folder Path",
            placeholder="Nhập đường dẫn lưu file...",
            value=load_path(),
            scale=4
        )
        save_btn = gr.Button("💾 Save Path", scale=1)


    # Generate button
    generate_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg", elem_classes=["generate-btn"])

    # Output
    with gr.Column(elem_classes=["output-card"]):
        gr.HTML('<div class="section-title">🔈 Output</div>')
        with gr.Row():
            output_audio = gr.Audio(label="Generated Speech", type="numpy", scale=2, interactive=False)
            status_text = gr.Textbox(label="Status", lines=2, scale=1)
    with gr.Row():
        save_audio_btn = gr.Button("💾 Lưu audio về máy", variant="secondary")
        saved_file = gr.File(label="File đã lưu", interactive=False)

    clear_btn.click(fn=lambda: "", outputs=[text_input])
    ref_dropdown.change(fn=lambda x: gr.update(value=x), inputs=[ref_dropdown], outputs=[ref_audio])
    # Khi bấm X ở audio, reset dropdown để lần chọn lại cùng file vẫn trigger update.
    ref_audio.clear(fn=lambda: None, outputs=[ref_dropdown])

    # Disable slider exaggeration khi chọn profile cụ thể (profile đã tự set exaggeration)
    # Enable lại khi chọn Custom (để user tự điều chỉnh)
    model_emotion.change(
        fn=_update_model_emotion_controls,
        inputs=[model_emotion],
        outputs=[exaggeration, ui_cfg_weight, ui_temperature, ui_top_p],
    )

    generate_btn.click(
        fn=_generate_speech,
        inputs=[text_input, language, ref_audio, tts_mode, 
                emotional_profile, exaggeration, model_emotion,
                ai_speed, ui_pitch_shift, ui_cfg_weight, ui_temperature, ui_top_p, ui_repetition_penalty],
        outputs=[output_audio, status_text]
    )

    # Thiết lập sự kiện khi bấm nút Save
    save_btn.click(fn=save_path, inputs=folder_input, outputs=status_text)

    # Voice Profile Builder — truyền giá trị exaggeration hiện tại sang builder
    build_profile_btn.click(
        fn=_run_build_voice_profile,
        inputs=[exaggeration],
        outputs=[build_profile_output],
    )
    copy_profile_btn.click(
        fn=_run_copy_profile_to_model,
        inputs=[],
        outputs=[build_profile_output],
    )
    save_audio_btn.click(
        fn=save_generated_audio,
        inputs=[output_audio, text_input, folder_input, numeric_input],
        outputs=[status_text, saved_file, numeric_input],
    )
    demo.load(
        fn=_update_model_emotion_controls,
        inputs=[model_emotion],
        outputs=[exaggeration, ui_cfg_weight, ui_temperature, ui_top_p],
    )
    demo.load(fn=lambda: 1, outputs=[numeric_input])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
