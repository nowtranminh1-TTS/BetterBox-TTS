from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """
    Cấu hình cho T3 (Token-To-Token) model.
    
    LƯU Ý QUAN TRỌNG:
    - Các thông số về token ID và vocab size PHẢI khớp với tokenizer đã train
    - Thay đổi các giá trị cố định (start/stop token, vocab size) sẽ khiến model
      hoạt động sai hoặc crash vì không tương thích với weights đã load
    """
    
    # ============================================
    # TEXT TOKENS - KHÔNG THAY ĐỔI (cố định theo tokenizer)
    # ============================================
    # Token đánh dấu bắt đầu/kết thúc chuỗi văn bản
    # Giá trị cố định: KHÔNG ĐƯỢC THAY ĐỔI - phải khớp với tokenizer
    start_text_token = 255   # Token bắt đầu câu (Start-of-Text)
    # Token 0 = stop_text_token, cũng dùng như <pad> cho CFG unconditional branch
    # Trong CFG: batch[1] được fill bằng token 0 để tạo "unconditional" signal
    stop_text_token = 0      # Token kết thúc câu (End-of-Text) / Pad token
    
    # Kích thước từ điển token văn bản 
    # - 704: cho model tiếng Anh/monolingual
    # - 2549: cho model đa ngôn ngữ (multilingual) có tiếng Việt
    # Giá trị cố định: PHẢI khớp với kích thước embedding trong weights đã train
    text_tokens_dict_size = 2549
    
    # Số token văn bản tối đa cho 1 inference
    # - Tăng: cho phép xử lý câu dài hơn (tối đa ~2048, giới hạn bởi pos emb và VRAM)
    # - Giảm: inference nhanh hơn, tiết kiệm VRAM, nhưng cắt câu dài
    # Khuyến nghị: 1024-2048. >2048 cần train lại position embeddings
    max_text_tokens = 2048
    
    # ============================================
    # SPEECH TOKENS - KHÔNG THAY ĐỔI (cố định theo S3 tokenizer)
    # ============================================
    # Token đánh dấu bắt đầu/kết thúc chuỗi speech
    # Giá trị cố định: KHÔNG ĐƯỢC THAY ĐỔI - phải khớp với S3 tokenizer
    start_speech_token = 6561   # Token bắt đầu đoạn âm thanh
    stop_speech_token = 6562    # Token kết thúc đoạn âm thanh
    
    # Kích thước từ điển token âm thanh (S3 tokenizer vocab)
    # Giá trị cố định: 8194 = 8192 codebook + 2 special tokens (start/stop)
    speech_tokens_dict_size = 8194
    
    # Số token âm thanh tối đa sinh ra = độ dài âm thanh tối đa
    # - 1 token ≈ 20-25ms (tùy S3 tokenizer hop length)
    # - 4096 tokens ≈ 80-100 giây âm thanh
    # - Tăng: cho phép sinh audio dài hơn, nhưng chậm hơn, tốn VRAM
    # - Giảm: inference nhanh hơn, nhưng nhiều chữ không biết đọc thì đọc sai
    # Khuyến nghị: 2048-4096 cho TTS thông thường -> dài hơn phải train thêm
    max_speech_tokens = 4096
    
    # ============================================
    # MODEL ARCHITECTURE - KHÔNG THAY ĐỔI (đã train cố định)
    # ============================================
    # Tên cấu hình Llama backbone
    # "Llama_520M": model 520M parameters (24 layers, 1024 hidden dim)
    # Giá trị cố định: PHẢI khớp với architecture weights đã train
    llama_config_name = "Llama_520M"
    
    # Loại positional embedding
    # "learned": học position embeddings (tốt cho TTS với độ dài biến đổi)
    # "rope" hoặc "alibi": alternative, nhưng model này dùng learned
    input_pos_emb = "learned"
    
    # ============================================
    # CONDITIONING - CÓ THỂ TINH CHỈNH (trong giới hạn)
    # ============================================
    # Độ dài prompt speech tokens cho conditioning giọng nói
    # đây chính là độ dài đầu vào của giọng clone
    # càng cao thì càng chiếm ram
    # 150 tokens  ≈  3 giây  (50 tokens/s)
    # 2000 tokens  ≈ 40 giây
    # 4050 tokens  ≈ 81 giây  ← giới hạn an toàn (speech_pos_emb trained tới 4096 - là 82 giây tối đa)
    speech_cond_prompt_len = 4050
    
    # Loại encoder cho speaker embedding
    # "voice_encoder": dùng pretrained voice encoder (Resemble AI)
    # Giá trị cố định: phải khớp với checkpoint đã train
    encoder_type = "voice_encoder"
    
    # Kích thước vector speaker embedding
    # 256: chuẩn cho voice encoder, đủ biểu diễn đặc trưng giọng nói
    # Giá trị cố định: phải khớp với speaker embed layer trong T3CondEnc
    speaker_embed_size = 256
    
    # Sử dụng Perceiver Resampler để nén conditioning
    # True: giảm chiều dài conditioning, inference nhanh hơn, VRAM ít hơn
    # False: giữ nguyên chiều dài, chậm hơn nhưng có thể chi tiết hơn
    # Có thể toggle: nhưng weights đã train với True nên để nguyên
    use_perceiver_resampler = True
    
    # Bật emotion adversarial conditioning
    # True: cho phép điều khiển độ biểu cảm giọng nói qua exaggeration parameter
    # False: không dùng emotion conditioning, giọng trung tính hơn
    # Có thể toggle: nhưng model đã train với emotion_adv=True
    emotion_adv = True

    @property
    def n_channels(self):
        """Trả về hidden size của Llama backbone (từ LLAMA_CONFIGS)"""
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @classmethod
    def multilingual(cls):
        """
        Tạo config cho model đa ngôn ngữ với vocab mở rộng.
        
        text_tokens_dict_size = 2549 bao gồm:
        - Vocab cơ bản (704 tokens)
        - Ký tự đặc biệt tiếng Việt (dấu thanh, nguyên âm đôi/ba)
        - Các ngôn ngữ khác trong tập train đa ngôn ngữ
        
        Lưu ý: Phải dùng checkpoint đã train cho multilingual 
        (t3_ml24ls_v2.safetensors), không dùng checkpoint monolingual.
        """
        config = cls()
        config.text_tokens_dict_size = 2549  # Expanded vocab for multilingual + Vietnamese
        return config
