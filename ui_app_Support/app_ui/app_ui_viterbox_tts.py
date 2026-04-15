import gradio as gr
from viterbox.AI_emotion_config import get_model_emotion_choices


def viterbox_UI_advance_AI_config():
    with gr.Row():
        model_emotion = gr.Dropdown(
            choices=get_model_emotion_choices(),
            value="AI-precision",
            label="🧠 Model AI Viterbox TTS Profile - for AI input",
            info="Profile cảm xúc từ tham số model AI (override exaggeration + cfg + temp + top_p khi chọn)",
        )

    with gr.Accordion("🧪 Advanced AI Viterbox TTS Parameters - for AI input", open=False):
        tts_mode = gr.Radio(
            choices=[("TTS normal", "normal"), ("TTS advance", "advance")],
            value="normal",
            label="Vitterbox TTS Mode",
            info="Normal: theo câu | Advance: theo từng từ",
        )
        gr.HTML(
            '<div style="font-size:0.82rem; color:#94a3b8; margin-bottom:0.5rem;">'
            "Tham số điều khiển trực tiếp model AI. Khi chọn Model Emotion Profile khác Default, "
            "các slider CFG/Temp/Top-P sẽ bị override bởi profile đó."
            "</div>"
        )

        exaggeration = gr.Slider(
            0,
            2,
            2,
            step=0.1,
            label="exaggeration - emotion - for AI input",
            info="cảm xúc. 0: âm đuôi cảm xúc mạnh, 2: âm đuôi mượt hơn",
            interactive=True,
        )

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

            ui_repetition_penalty = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="🔁 Repetition Penalty",
                info=">1.0 tránh lặp token, quá cao sẽ cứng",
            )

    return (
        model_emotion,
        tts_mode,
        exaggeration,
        ui_cfg_weight,
        ui_temperature,
        ui_top_p,
        ui_repetition_penalty,
    )


def viterbox_bind_advanced_ai_config_actions(
    demo,
    model_emotion,
    exaggeration,
    ui_cfg_weight,
    ui_temperature,
    ui_top_p,
    ui_repetition_penalty,
    update_model_emotion_controls_fn,
):
    """Bind events for Advanced AI Parameters controls."""
    model_emotion.change(
        fn=update_model_emotion_controls_fn,
        inputs=[model_emotion],
        outputs=[exaggeration, ui_cfg_weight, ui_temperature, ui_top_p, ui_repetition_penalty],
    )

    demo.load(
        fn=update_model_emotion_controls_fn,
        inputs=[model_emotion],
        outputs=[exaggeration, ui_cfg_weight, ui_temperature, ui_top_p, ui_repetition_penalty],
    )


def viterbox_UI_build_voice_profile():
    """Build UI for Voice Profile Builder."""
    with gr.Accordion("🧠 Voice Profile Builder viterbox tts", open=False):
        with gr.Column(elem_classes=["card"]):
            gr.HTML(
                '<div style="font-size:0.9rem; color:#ffffff; margin-bottom:0.5rem;">'
                "Gộp audio trong viterbox/pretrained/ → tạo conditioning tối ưu → lưu vào viterbox/output-profile/. "
                "Nhấn Copy để dùng ngay làm default (cần restart app)."
                "</div>"
            )

            with gr.Row():
                build_profile_btn = gr.Button(
                    "🧠 Build Voice Profile",
                    variant="primary",
                    size="sm",
                    scale=3,
                )
                copy_profile_btn = gr.Button(
                    "📋 Copy → modelViterboxLocal",
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
    return build_profile_btn, copy_profile_btn, build_profile_output


def viterbox_bind_voice_profile_actions(
    build_profile_btn,
    copy_profile_btn,
    build_profile_output,
    exaggeration,
    run_build_voice_profile_fn,
    run_copy_profile_to_model_fn,
):
    """Bind events for Voice Profile Builder controls."""
    build_profile_btn.click(
        fn=run_build_voice_profile_fn,
        inputs=[exaggeration],
        outputs=[build_profile_output],
    )
    copy_profile_btn.click(
        fn=run_copy_profile_to_model_fn,
        inputs=[],
        outputs=[build_profile_output],
    )