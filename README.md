# 🎙️ Betterbox TTS

### app base trên Vitterbox tts: https://github.com/iamdinhthuan/viterbox-tts

## một số tính năng mới 
- 1. thêm điều chỉnh tốc độ trực tiếp trong model s3gen
- 2. bỏ giới hạn chỉ 6 giây audio mẫu - giờ audio mẫu lên tối đa 80 giây - tăng độ chính xác khi TTS, nhưng chờ lâu
- 3. thêm tính năng 'Voice Profile Builder'. tối đa 26 phút audio -> giống với audio prompt, nhưng chỉ làm 1 lần, không ảnh hưởng hiệu năng khi TTS như audio prompt
- 4. thêm EQ với thư viện pedalboard và pydub
- 5. lưu vị trí download
- 6. thêm tính năng pitch với thư viện pedalboard và pydub
- 7. fix code để tránh lỗi tích lũy khi gen audio dài 
- 8. fix lỗi nuốt chữ.
- 9. thêm runApp.bat - sau khi có venv và cài thư viện với venv, chỉ cần click file này là chạy app
- 10. tính năng 'Thứ tự:', cho phép file audio có thêm mục số ở đầu tên

# tải model và đưa vào folder 'modelTTSLocal' (tránh chép đè file conds.pt - đây là file config để voice mẫu hiện tại chạy chính xác)
https://huggingface.co/dolly-vn/viterbox/tree/main

# demo app:
https://github.com/user-attachments/assets/9ff920e3-6779-49c2-b61f-67a841295635


## 📦 Cài đặt

### Yêu cầu hệ thống

- **Python**: 3.10+
- **CUDA**: 11.8+ (khuyến nghị)
- **RAM**: 8GB+
- **VRAM**: 6GB+ (GPU) (10GB+ nếu xài từ 20 phút chức năng 'Voice Profile Builder')

### Cài đặt từ source

```bash
# Clone repo
cd viterbox

# Tạo virtual environment (khuyến nghị) - tạo trong thư mục viterbox
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Cài đặt với pip

```bash
pip install -e .
```

---

## 🚀 Sử dụng

### 1. Giao diện Web (Gradio)

```bash
python app.py
```

Mở trình duyệt tại `http://localhost:7860`

hoặc sau khi có venv, thì chạy file 'runApp.bat' - file tự động bật venv và chạy

---

## 🎛️ Tham số

| Tham số | Mô tả | Giá trị | Mặc định |
|---------|-------|---------|----------|
| `text` | Văn bản cần đọc | string | (bắt buộc) |
| `language` | Mã ngôn ngữ | `"vi"`, `"en"` | `"vi"` |
| `audio_prompt` | Audio mẫu cho voice cloning | path/tensor | `None` |
| `exaggeration` | Mức độ biểu cảm | 0.0 - 2.0 | 0.5 |
| `cfg_weight` | Độ bám sát giọng mẫu | 0.0 - 1.0 | 0.5 |
| `temperature` | Độ ngẫu nhiên/sáng tạo | 0.1 - 1.0 | 0.8 |
| `top_p` | Top-p sampling | 0.0 - 1.0 | 0.9 |
| `repetition_penalty` | Phạt lặp từ | 1.0 - 2.0 | 1.2 |
| `sentence_pause_ms` | Thời gian ngắt giữa câu | 0 - 2000 | 500 |
| `crossfade_ms` | Thời gian crossfade | 0 - 100 | 50 |

### Giải thích tham số

- **exaggeration**: Tăng để giọng biểu cảm hơn, giảm để giọng trầm tĩnh hơn
- **cfg_weight**: Tăng để giọng giống mẫu hơn, giảm để tự nhiên hơn
- **temperature**: Tăng để giọng đa dạng hơn, giảm để ổn định hơn
- **sentence_pause_ms**: Thời gian nghỉ giữa các câu (hữu ích cho văn bản dài)
- **Pitch Shift (semitones)**: cao độ (tone) của giọng đầu ra

---

## 📁 Cấu trúc dự án

```
viterbox/
├── app.py                  # Gradio Web UI
├── inference.py            # CLI inference script
├── requirements.txt        # Dependencies
├── pyproject.toml          # Package config
├── README.md
├── wavs/                   # Thư mục chứa giọng mẫu
│   └── *.wav
├── modelTTSLocal/          # Thư mục chứa model local
├── output-profile/         # Thư mục chứa file kết quả của Voice Profile
├── pretrained/             # Thư mục chứa audio + text cho Voice Profile
└── viterbox/               # Core library
    ├── __init__.py
    ├── tts.py              # Main Viterbox class
    └── models/             # Model components
        ├── t3/             # T3 Text-to-Token model
        ├── s3gen/          # S3Gen vocoder
        ├── s3tokenizer/    # Speech tokenizer
        ├── voice_encoder/  # Speaker encoder
        └── tokenizers/     # Text tokenizer
```

---

## 🔧 Model Files

source được fix lại để model chạy local, không tự tải về. 
model sau khi download xong, thì đặt trong folder 'modelTTSLocal'
sau đó, để nâng cao chất lượng TTS đầu ra, cần copy đè file 'conds.pt' trong folder 'output-profile' vào trong folder 'modelTTSLocal'

Model được host trên HuggingFace Hub: [`dolly-vn/viterbox`](https://huggingface.co/dolly-vn/viterbox)

| File | Mô tả | Kích thước |
|------|-------|------------|
| `t3_ml24ls_v2.safetensors` | T3 model (fine-tuned) | ~2GB |
| `s3gen.pt` | S3Gen vocoder | ~1GB |
| `ve.pt` | Voice Encoder | ~20MB |
| `tokenizer_vi_expanded.json` | Tokenizer với vocab tiếng Việt | ~50KB |
| `conds.pt` | Default voice conditioning | ~1MB |

---

## 🧠 Lưu ý khi tạo Voice Profile

Áp dụng cho dữ liệu trong folder `pretrained/`:

1. **Chỉ dùng 1 giọng duy nhất**  
   Trộn nhiều giọng sẽ làm output không ổn định.
2. **Nên có file text đi kèm từng audio**  
   Đặt cùng tên, ví dụ: `clip1.mp3` + `clip1.txt`.  
   App dùng text để chọn window 80s đa dạng âm vị hơn.
3. **Độ dài audio tối đa 26 phút - nên để 25 phút thôi**  
   `speaker_emb` và `x-vector` được tính từ toàn bộ audio (không cắt 80s).  
   Acoustic context (Perceiver Average) tổng hợp từ tối đa 20 cửa sổ x 80s (~26 phút).
4. trong folder 'pretrained' đã để sẵn audio và text (tạo bởi AI) để chạy chức năng này

Audio prompt khi chạy app nên cùng giọng với audio đã dùng để build profile.  
Kết quả build là file `conds.pt` trong `output-profile/`.  
Dùng nút `Copy -> modelTTSLocal` để app dùng ngay (cần restart app).

---

## ⚠️ Lưu ý

- **Audio mẫu**: Nên sử dụng audio sạch, không nhiễu, 3-10 giây
- **VRAM**: Model cần ~6GB VRAM, nếu không đủ có thể dùng CPU (chậm hơn)
- **Văn bản**: Hỗ trợ tốt nhất với văn bản có dấu đầy đủ

---

## 🙏 Credits

- **Base Model**: [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI
- **Datasets**: ViVoice, PhoAudiobook, Dolly-Audio
- **Fine-tuning**: [Dolly VN](https://github.com/dolly-vn) - Speech Team @ [ContextBoxAI](https://contextbox.ai)

---

## 📄 License

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0)

- ✅ Được sử dụng cho mục đích **phi thương mại**
- ✅ Được chia sẻ, sửa đổi với ghi nguồn
- ❌ **KHÔNG** được sử dụng cho mục đích thương mại
- ❌ **KHÔNG** được sử dụng cho mục đích xấu xa
- file audio là người thật đọc, mình lấy từ tiktok.

Liên hệ thương mại: [contextbox.ai](https://contextbox.ai)

---

**Made with ❤️ by [Dolly VN](https://github.com/dolly-vn) @ [ContextBoxAI](https://contextbox.ai)**

[⬆ Về đầu trang](#️-betterbox-tts)
