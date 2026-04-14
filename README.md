# 🎙️ Betterbox TTS - V2

### app base trên Vitterbox tts: https://github.com/iamdinhthuan/viterbox-tts

## một số tính năng mới - chung: 
- 1. bổ sung tùy chọn model Omnivoice hoặc Viterbox ngoài UI. click chọn là chạy.
- 2. fix bug UI
- Note: hiện model Omnivoice còn thô sơ, chưa có chức năng ngắt câu theo dấu câu.  

## một số tính năng mới - cho model Viterbox
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
- 11. tính năng 'advance TTS' - cho câu rất chính xác, nhưng nghe như robot 😁

# 🔧 tải model viterbox TTS và đưa vào folder 'viterbox/modelViterboxLocal' 
(tránh chép đè file conds.pt - đây là file config để voice mẫu hiện tại chạy chính xác)
https://huggingface.co/dolly-vn/viterbox/tree/main


# 🔧 tải model Omnivoice và đưa vào folder 'OmniVoice/modelOmniLocal' 
https://huggingface.co/k2-fsa/OmniVoice/tree/main



## 📁 Cấu trúc dự án

```
viterbox-TTS=GPU/
├── app.py                  # Gradio Web UI
├── inference.py            # CLI inference script
└── general/                # Core library
    ├── general/requirements.txt    # Dependencies (Windows/Linux)
    ├── general/requirements-mac.txt# Dependencies (macOS)
    ├── config_path.txt     # lưu đường dẫn folder download audio
    └── EQ_emotion_config/  # chứa các file config âm thanh bằng EQ
├── pyproject.toml          # Package config
├── README.md
├── wavs/                   # Thư mục chứa giọng mẫu
│   └── *.wav
├── OmniVoice/              # folder với model OmniVoice + file inference
│   ├── modelOmniLocal/     # Thư mục chứa model local OmniVoice
│   ├── omnivoice/          # Model components OmniVoice
│   └── omnivoice_inference/# Folder chứa phần suy luận của OmniVoice
│       ├── ttsOmni.py      # File suy luận cho Omnivoice
└── viterbox/               # Core library
    ├── modelViterboxLocal/ # Thư mục chứa model local Viterbox(base trên Chatterbox)
    ├── output-profile/     # Thư mục chứa file kết quả của Voice Profile
    ├── pretrained/         # Thư mục chứa audio + text cho Voice Profile
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

## 📦 Cài đặt - cách cài đặt venv và thư viện ở bản V2 thì y như cũ

### Yêu cầu hệ thống

- **Python**: 3.10+
- **CUDA**: 11.8+ (khuyến nghị)
- **RAM**: 8GB+
- **VRAM**: 6GB+ (GPU) - 8GB nếu xài Omnivoice 
(10GB+ nếu xài từ 20 phút chức năng 'Voice Profile Builder' của Viterbox)

### Cài đặt từ source

```bash
# Clone repo
cd viterbox

# Tạo virtual environment (khuyến nghị) - tạo trong thư mục viterbox
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r general/requirements.txt
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

## 👉👉👉 Thông số dành cho model Omnivoice
- xem trên: https://huggingface.co/k2-fsa/OmniVoice
- cấu hình khi chạy Omnivoice: 7GB VRAM - nếu thiếu sẽ tự load một phần lên RAM, nhưng sẽ chậm

## 👉👉👉 Thông số dành cho model viterbox tts

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


## 🧠 Lưu ý khi tạo Voice Profile

Áp dụng cho dữ liệu trong folder `viterbox/pretrained/`:

1. **Chỉ dùng 1 giọng duy nhất**  
   Trộn nhiều giọng sẽ làm output không ổn định.
2. **Nên có file text đi kèm từng audio**  
   Đặt cùng tên, ví dụ: `clip1.mp3` + `clip1.txt`.  
   App dùng text để chọn window 80s đa dạng âm vị hơn.
3. **Độ dài audio tối đa 26 phút - nên để 25 phút thôi**  
   `speaker_emb` và `x-vector` được tính từ toàn bộ audio (không cắt 80s).  
   Acoustic context (Perceiver Average) tổng hợp từ tối đa 20 cửa sổ x 80s (~26 phút).
4. trong folder 'viterbox/pretrained' đã để sẵn audio và text (tạo bởi AI) để chạy chức năng này

Audio prompt khi chạy app nên cùng giọng với audio đã dùng để build profile.  
Kết quả build là file `conds.pt` trong `viterbox/output-profile/`.  
Dùng nút `Copy -> modelViterboxLocal` để app dùng ngay (cần restart app), file sẽ được copy vào `viterbox/modelViterboxLocal/`.

---

## DEMO APP 


https://github.com/user-attachments/assets/9d761cfb-b1de-46c2-bf74-32b09c692403



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
