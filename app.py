"""
Viterbox - Gradio Web Interface
"""
# Set HF Hub env vars BEFORE importing transformers to disable warnings
import os

# Disable telemetry: Prevent Hugging Face from sending usage statistics/analytics
# Điều này tránh các request ngầm đến HF Hub để báo cáo dữ liệu sử dụng
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Force offline mode: Không gọi API đến HF Hub, chỉ dùng model local
# Điều này tránh warning "unauthenticated requests" vì không còn request nào được gửi đi
os.environ["HF_HUB_OFFLINE"] = "1"

# Set dummy token để tránh warning "unauthenticated requests"
# Vì đang ở offline mode, token này sẽ không được sử dụng cho bất kỳ request nào
# nhưng sẽ làm hài lòng auth check của huggingface_hub
os.environ["HF_TOKEN"] = "dummy"

# Disable symlink warning: Tránh warning về việc Windows không hỗ trợ symlinks tốt
# (thường xuất hiện khi HF Hub cố tạo symlink cho cache files)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import warnings
import gradio as gr

warnings.filterwarnings('ignore')

import tempfile

os.environ["GRADIO_TEMP_DIR"] = tempfile.gettempdir() + "/my_gradio_tmp"
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# Clear temp folder on startup (không crash nếu lỗi)
try:
    import shutil
    temp_dir = os.environ["GRADIO_TEMP_DIR"]
    if os.path.exists(temp_dir):
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception:
                pass  # Bỏ qua file đang bị lock
except Exception:
    pass  # Không crash app nếu có lỗi

from pathlib import Path
from typing import cast
from gradio.components.textbox import InputHTMLAttributes
from OmniVoice.omnivoice_inference.ttsOmni import generate_speech_omni
from general.EQ_emotion_config.eq_emotional_profiles import list_emotional_profiles, get_profile_description
from viterbox.pretrain_voice_builder import build_voice_profile, copy_profile_to_model, PRETRAINED_DIR, OUTPUT_DIR, MODEL_DIR
from viterbox.tts_generate_speech import generate_speech_viterbox
from ui_app_Support.app_ui.app_ui_viterbox_tts import (
    viterbox_UI_advance_AI_config,
    viterbox_bind_advanced_ai_config_actions,
    viterbox_UI_build_voice_profile,
    viterbox_bind_voice_profile_actions,
)
from ui_app_Support.app_support.app_model_management import (
    configure_device,
    ensure_active_model,
    get_model_viterbox,
    get_omni_model,
)
from ui_app_Support.app_support.app_support import (
    CSS, APP_INIT_JS,
    list_voices, get_default_voice, get_wavs_dir,
    save_path, load_path,
    save_generated_audio_and_srt,
    run_build_voice_profile, run_copy_profile_to_model,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

configure_device(DEVICE)

# ── Wrapper functions (inject MODEL + dirs into app_support functions) ─────────

def _run_build_voice_profile(exaggeration_val):
    model = get_model_viterbox()
    return run_build_voice_profile(model, PRETRAINED_DIR, OUTPUT_DIR, build_voice_profile, exaggeration_val)

def _run_copy_profile_to_model():
    return run_copy_profile_to_model(OUTPUT_DIR, MODEL_DIR, copy_profile_to_model)

# Thuộc tính HTML gắn trực tiếp lên ô nhập (Gradio ≥ 4) — tắt spellcheck trình duyệt cho tiếng Việt.
TTS_TEXT_HTML_ATTRS = cast(
    InputHTMLAttributes,
    {
        "spellcheck": False,
        "autocorrect": "off",
        "autocapitalize": "off",
        "autocomplete": "off",
        "lang": "vi",
    },
)


def _update_model_emotion_controls(profile):
    is_custom = profile == "AI-custom"
    return (
        gr.update(interactive=is_custom),
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
    js=APP_INIT_JS
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
                lines=5,
                elem_id="main-text-input",
                html_attributes=TTS_TEXT_HTML_ATTRS,
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")

            # ── Voice Profile Builder for viterbox tts ─────────────────────────────────────
            with gr.Column(visible=False) as viterbox_profile_container:
                (
                    build_profile_btn,
                    copy_profile_btn,
                    build_profile_output,
                ) = viterbox_UI_build_voice_profile()

            # nhập thứ tự audio để save
            with gr.Row():
                model_choice = gr.Radio(
                    choices=[("Viterbox", "viterbox"), ("Omni", "omni")],
                    value="omni",
                    label="Model",
                    info="Chọn model để inference",
                )

        # Right - Voice & Settings
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="section-title">🎤 Reference Voice</div>')
            gr.HTML('''<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-left: 4px solid #4fc3f7; border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 13px; color: #ffffff; line-height: 1.5;">
                <span style="color: #4fc3f7; font-weight: bold;">💡 Tip:</span> Để OmniVoice cho kết quả chính xác nhất, hãy đặt file <code style="background: rgba(255,255,255,0.15); padding: 2px 6px; border-radius: 4px; color: #ffffff;">.wav</code> và <code style="background: rgba(255,255,255,0.15); padding: 2px 6px; border-radius: 4px; color: #fff;">.txt</code> cùng tên trong folder wavs/
            </div>''')      
            wav_files = list_voices()
            default_voice = get_default_voice(wav_files)
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

                    # ── Advanced AI Parameters for viterbox tts ─────────────────────────────────────
                    with gr.Column(visible=False) as viterbox_adv_config_container:
                        (
                            model_emotion,
                            tts_mode,
                            exaggeration,
                            ui_cfg_weight,
                            ui_temperature,
                            ui_top_p,
                            ui_repetition_penalty,
                        ) = viterbox_UI_advance_AI_config()

    # Toggle Viterbox UI based on model selection
    def toggle_viterbox_ui(choice):
        is_viterbox = (choice == "viterbox")
        return gr.update(visible=is_viterbox), gr.update(visible=is_viterbox)

    model_choice.change(
        fn=toggle_viterbox_ui,
        inputs=[model_choice],
        outputs=[viterbox_profile_container, viterbox_adv_config_container]
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
    generate_btn = gr.Button("🔊 Generate Speech + SRT audio", variant="primary", size="lg", elem_classes=["generate-btn"])

    # Output
    with gr.Column(elem_classes=["output-card"]):
        gr.HTML('<div class="section-title">🔈 Output</div>')
        with gr.Row():
            output_audio = gr.Audio(label="Generated Speech", type="numpy", scale=2, interactive=False)
            status_text = gr.Textbox(label="Status", lines=2, scale=1)
    with gr.Row():
        save_audio_btn = gr.Button("💾 Lưu audio + SRT về máy", variant="secondary")

        with gr.Column(scale=1, elem_classes=["card"]):
            saved_file = gr.File(label="File đã lưu", interactive=False)
            srt_file = gr.File(label="SRT File", interactive=False, visible=True)

    clear_btn.click(fn=lambda: "", outputs=[text_input])
    ref_dropdown.change(fn=lambda x: gr.update(value=x), inputs=[ref_dropdown], outputs=[ref_audio])
    # Khi bấm X ở audio, reset dropdown để lần chọn lại cùng file vẫn trigger update.
    ref_audio.clear(fn=lambda: None, outputs=[ref_dropdown])

    def generate_speech_fn(data):
        mc = data[model_choice]
        try:
            switched = ensure_active_model(mc)
        except Exception as e:
            return None, f"❌ Model switch error: {str(e)}"

        if mc == "omni":
            # inference với model omni - chỉ dùng các tham số cần thiết cho Omni
            try:
                omni_model = get_omni_model()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Omni load error: {str(e)}"

            # Lấy ref_text và đường dẫn gốc từ folder wavs/ (thay vì dùng temp file của Gradio)
            ref_audio_temp_path = data[ref_audio]

            # Lấy tên file (không extension) từ đường dẫn temp của Gradio
            audio_filename = Path(ref_audio_temp_path).stem

            # Tìm file gốc trong folder wavs/
            wavs_dir = get_wavs_dir()
            ref_audio_path = wavs_dir / f"{audio_filename}.wav"
            ref_text_path = wavs_dir / f"{audio_filename}.txt"

            # Kiểm tra file gốc có tồn tại không
            if not ref_audio_path.exists():
                # Fallback: dùng temp path nếu không tìm thấy trong wavs/
                ref_audio_path = Path(ref_audio_temp_path)

            # Lấy ref_text
            ref_text = None
            if ref_text_path.exists():
                try:
                    with open(ref_text_path, "r", encoding="utf-8") as f:
                        ref_text = f.read().strip()
                except Exception:
                    ref_text = None

            print(f"\n📁 wavs_dir: {wavs_dir}")
            print(f"📝 ref_text_path: {ref_text_path}")
            print(f"🎵 audio path (temp): {ref_audio_temp_path}")
            print(f"🎵 audio path (wavs): {ref_audio_path}")
            print(f"📄 ref_text: {'Found' if ref_text else 'None'}\n")

            audio_out, status, srtFileResult = generate_speech_omni(
                omni=omni_model,
                text=data[text_input],
                language=data[language],
                reference_audio=str(ref_audio_path),
                ref_text=ref_text,
                speed=data[ai_speed],
                pitch_shift=data[ui_pitch_shift],
            )
            if switched and status:
                status = f"🔁 Switched to Omni | {status}"
            return audio_out, status, srtFileResult
        else:
            # inference với model viterbox - dùng các tham số chi tiết cho Viterbox
            try:
                model = get_model_viterbox()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Viterbox load error: {str(e)}"

            audio_out, status, srtFileResult = generate_speech_viterbox(
                MODEL=model, 
                text=data[text_input], 
                language=data[language], 
                reference_audio=data[ref_audio], 
                tts_mode=data[tts_mode],
                emotional_profile=data[emotional_profile], 
                ui_exaggeration=data[exaggeration], 
                model_emotion_profile=data[model_emotion],
                ai_speed=data[ai_speed], 
                ui_cfg_weight=data[ui_cfg_weight], 
                ui_temperature=data[ui_temperature], 
                ui_top_p=data[ui_top_p], 
                ui_repetition_penalty=data[ui_repetition_penalty], 
                ui_pitch_shift=data[ui_pitch_shift],
            )
            if switched and status:
                status = f"🔁 Switched to Viterbox | {status}"

            return audio_out, status, srtFileResult

    viterbox_bind_advanced_ai_config_actions(
        demo=demo,
        model_emotion=model_emotion,
        exaggeration=exaggeration,
        ui_cfg_weight=ui_cfg_weight,
        ui_temperature=ui_temperature,
        ui_top_p=ui_top_p,
        ui_repetition_penalty=ui_repetition_penalty,
        update_model_emotion_controls_fn=_update_model_emotion_controls,
    )

    # Define separate input sets for each model for better maintainability
    inputs_omni = {
        model_choice, text_input, language, ref_audio, ai_speed, ui_pitch_shift
    }
    inputs_viterbox = {
        model_choice, text_input, language, ref_audio, tts_mode,
        emotional_profile, exaggeration, model_emotion,
        ai_speed, ui_pitch_shift, ui_cfg_weight, ui_temperature, ui_top_p, ui_repetition_penalty
    }

    generate_btn.click(
        fn=generate_speech_fn,
        inputs=inputs_omni | inputs_viterbox,  # Union of all necessary components
        outputs=[output_audio, status_text, srt_file]
    )

    # Thiết lập sự kiện khi bấm nút Save
    save_btn.click(fn=save_path, inputs=folder_input, outputs=status_text)

    # Voice Profile Builder
    viterbox_bind_voice_profile_actions(
        build_profile_btn=build_profile_btn,
        copy_profile_btn=copy_profile_btn,
        build_profile_output=build_profile_output,
        exaggeration=exaggeration,
        run_build_voice_profile_fn=_run_build_voice_profile,
        run_copy_profile_to_model_fn=_run_copy_profile_to_model,
    )

    save_audio_btn.click(
        fn=save_generated_audio_and_srt,
        inputs=[output_audio, text_input, folder_input, srt_file],
        outputs=[status_text, saved_file],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
