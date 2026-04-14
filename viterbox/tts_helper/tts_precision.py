"""
tts_precision.py — Bộ tiền xử lý text cực đoan cho TTS tiếng Việt
══════════════════════════════════════════════════════════════════════════════

FILE NÀY CHỨA HÀM `config_token_for_precision` và tất cả logic liên quan.
Được tách ra từ tts_support.py để dễ quản lý và mở rộng.

════════════════════════════════════════════════════════════════════════════
PHÂN TÍCH GỐC RỄ — TẠI SAO CÓ ÂM LẠ Ở ĐUÔI CÂU DÀI?
════════════════════════════════════════════════════════════════════════════

1. TOKENIZER LÀ BPE VỚI CHỈ 265 MERGES (chủ yếu tiếng Anh)
   - pre_tokenizer: Whitespace → tách theo space TRƯỚC, rồi BPE
   - Merges là: "th", "in", "the", "an", "er", "ing", "and"... → 100% tiếng Anh
   - Tiếng Việt KHÔNG CÓ merge nào → mỗi chữ bị bẻ thành TỪNG KÝ TỰ ĐƠN
   - VD: "xin" → [x, i, n] = 3 tokens; nhưng tiếng Anh "the" → 1 token
   - Hệ quả: câu tiếng Việt dài tạo ra dãy token DÀI GẤP 2-3 LẦN so với cùng nội dung tiếng Anh

2. T3 LÀ AUTOREGRESSIVE (LLaMA-based)
   - Token thứ N phụ thuộc vào TẤT CẢ token trước đó
   - Dãy token càng dài → ngữ cảnh càng loãng → xác suất tích lũy giảm
   - Sau ~500-600 speech tokens, model bắt đầu "trượt" distribution → sinh gibberish
   - Đây là vấn đề CỐ HỮU của autoregressive models, KHÔNG thể fix bằng training

3. VĂN BẢN TRƯỚC KHI VÀO TOKENIZER CÓ KÝ TỰ "RÁC"
   - clearText() chuyển dấu câu → ", " nhưng KHÔNG xóa ký tự ngoài vocab
   - Ký tự ngoài vocab → [UNK] (ID=1) → T3 không biết reader gì → noise
   - VD: emoji, ký tự đặc biệt, zero-width space (U+200B-200D) trong vocab!

4. UNICODE NORMALIZATION VẤN ĐỀ
   - Tiếng Việt có 2 cách biểu diễn Unicode:
     * NFC (Composed):  "ắ" = 1 codepoint (U+1EAF, ID ~vocab entry duy nhất)
     * NFD (Decomposed): "ắ" = "a" + "̆" + "́" = 3 codepoints (3 tokens riêng!)
   - Tokenizer KHÔNG có normalizer → nếu text là NFD, mỗi dấu thành token riêng
   - Đây là NGUYÊN NHÂN CHÍNH gây token dài bất thường cho tiếng Việt

════════════════════════════════════════════════════════════════════════════
GIẢI PHÁP IMPLEMENT TRONG FILE NÀY (thứ tự quan trọng)
════════════════════════════════════════════════════════════════════════════

Bước 0: Unicode NFC Normalization — quan trọng nhất, giảm 30-50% token count
Bước 1: Loại bỏ ký tự zero-width và invisible Unicode
Bước 2: Chuẩn hoá dấu câu → dấu chấm (.)
Bước 3: Lọc ký tự ngoài bảng vocab thực tế (đọc từ JSON, không hardcode)
Bước 4: Chuẩn hoá khoảng trắng
Bước 5: Tách từ quá dài (edge case: URL, mã sản phẩm)
Bước 6: Thêm boundary pause " . text . "
"""

import re
import unicodedata
import json
import os
from pathlib import Path
from typing import Optional, Set


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 0: Xây dựng bảng VOCAB thực tế từ tokenizer JSON
# ════════════════════════════════════════════════════════════════════════════
# Đọc trực tiếp từ file tokenizer → KHÔNG BAO GIỜ bị lệch với model
# Thay vì hardcode (dễ sai, dễ quên), ta dùng bảng thật.

def _load_vocab_charset(tokenizer_path: Optional[str] = None) -> Set[str]:
    """
    Đọc tokenizer JSON và trích xuất TẤT CẢ ký tự đơn (single-char tokens)
    có trong vocab. Đây là tập ký tự mà tokenizer THỰC SỰ "biết".

    Ký tự nằm ngoài tập này → tokenizer map về [UNK] → T3 sinh noise.

    Returns:
        Set[str]: tập ký tự đơn hợp lệ
    """
    if tokenizer_path is None:
        # Tìm tokenizer_vi_expanded.json tự động
        # Cấu trúc: viterbox/tts_helper/tts_precision.py → lên 1 cấp → viterbox/modelViterboxLocal/
        _this_dir = Path(__file__).parent
        tokenizer_path = str(_this_dir.parent / "modelViterboxLocal" / "tokenizer_vi_expanded.json")

    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})

        # Chỉ lấy ký tự đơn (len=1), bỏ qua special tokens ([], BPE merges, etc.)
        single_chars = set()
        for token_str, token_id in vocab.items():
            if len(token_str) == 1:
                single_chars.add(token_str)

        # Luôn đảm bảo có khoảng trắng (dù tokenizer dùng [SPACE])
        single_chars.add(' ')

        return single_chars

    except Exception as e:
        print(f"  ⚠️ [tts_precision] Không đọc được tokenizer JSON: {e}")
        print(f"  ⚠️ Fallback về bảng ký tự mặc định (có thể thiếu)")
        # Fallback: bảng tối thiểu cho tiếng Việt + ASCII
        return set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:-/'\"()[]"
            # Vietnamese precomposed (NFC)
            "àáảãạăắằẳẵặâấầẩẫậ"
            "èéẻẽẹêếềểễệ"
            "ìíỉĩị"
            "òóỏõọôốồổỗộơớờởỡợ"
            "ùúủũụưứừửữự"
            "ỳýỷỹỵ"
            "đ"
            "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
            "ÈÉẺẼẸÊẾỀỂỄỆ"
            "ÌÍỈĨỊ"
            "ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
            "ÙÚỦŨỤƯỨỪỬỮỰ"
            "ỲÝỶỸỴ"
            "Đ"
        )


# ════════════════════════════════════════════════════════════════════════════
# Bảng ký tự hợp lệ — load 1 lần duy nhất khi import module
# ════════════════════════════════════════════════════════════════════════════
_VOCAB_CHARS: Set[str] = _load_vocab_charset()

# Ký tự cho phép đi qua filter (vocab + space)
# Bao gồm cả combining diacritics vì sau NFC normalize thì chúng không nên tồn tại,
# nhưng nếu có thì giữ lại để tránh hỏng chữ
_ALLOWED_CHARS = _VOCAB_CHARS.copy()

# ════════════════════════════════════════════════════════════════════════════
# Ký tự invisible cần xóa triệt để
# ════════════════════════════════════════════════════════════════════════════
# Các ký tự này CÓ MẶT trong vocab (VD: U+200B=ID2049, U+200C=ID2050, U+200D=ID2051)
# nhưng KHÔNG PHẢI ký tự phát âm → chỉ chiếm slot token mà không tạo âm thanh
# → bỏ đi để tiết kiệm token budget cho T3

_INVISIBLE_RE = re.compile(
    '['
    '\u200b'    # zero-width space (ID 2049 trong vocab!)
    '\u200c'    # zero-width non-joiner (ID 2050)
    '\u200d'    # zero-width joiner (ID 2051)
    '\u200e'    # left-to-right mark
    '\u200f'    # right-to-left mark
    '\u2060'    # word joiner (ID 2102)
    '\u2061'    # function application (ID 2103)
    '\u2062'    # invisible times (ID 2104)
    '\u2063'    # invisible separator (ID 2105)
    '\u2064'    # invisible plus (ID 2106)
    '\u206a-\u206f'  # deprecated format chars (ID 2111-2116)
    '\ufeff'    # byte order mark
    '\ufffe'    # not a character
    '\uffff'    # not a character
    '\u00ad'    # soft hyphen (ID 342 trong vocab — không phát âm)
    '\u00a0'    # non-breaking space → sẽ thay bằng space thường
    ']+', re.UNICODE
)

# ─── Dấu câu gốc tiếng Anh mà tokenizer BIẾT (có trong vocab) ──────────
# Tokenizer có: . (9), , (7), ! (3), ? (13), ; (12), : (11), - (8), / (10), ' (4)
# Model được TRAIN nhiều nhất với dấu "." vì dataset chủ yếu là narrative text.
# Các dấu khác (! ? ；？ ！ ，) ít gặp → model có thể sinh âm điệu bất ngờ.
_PUNCT_TO_PERIOD = str.maketrans({
    '!': '.', '？': '.', '！': '.', '¡': '.',
    '¿': '.', '？': '.',
    '。': '.', '．': '.',
    ';': ',', '；': ',',  # dấu chấm phẩy → dấu phẩy (pause nhẹ hơn period)
    '：': ',', ':': ',',  # dấu hai chấm → dấu phẩy
    '—': ',', '–': ',', '―': ',',  # em dash, en dash → dấu phẩy
    '…': '.',  # ellipsis → period
})


# ════════════════════════════════════════════════════════════════════════════
# HÀM CHÍNH
# ════════════════════════════════════════════════════════════════════════════

def config_token_for_precision(
    text: str,
    *,
    # ── Bước xử lý (tất cả mặc định BẬT — tắt riêng khi cần debug) ──
    unicode_nfc: bool = True,           # Bước 0: NFC normalize (QUAN TRỌNG NHẤT)
    remove_invisible: bool = True,      # Bước 1: Xóa zero-width chars
    normalize_punctuation: bool = True, # Bước 2: Chuẩn hoá dấu câu → period/comma
    filter_unk_chars: bool = True,      # Bước 3: Lọc ký tự ngoài vocab
    normalize_whitespace: bool = True,  # Bước 4: Dọn khoảng trắng
    split_long_words: bool = True,      # Bước 5: Tách từ dài > threshold
    long_word_threshold: int = 12,      # Ngưỡng ký tự (VN đơn âm tiết → 12 là đã rất dài)
    add_boundary_pause: bool = True,    # Bước 6: Thêm " . text . "
    debug: bool = False,                # In log từng bước
) -> str:
    """
    Tiền xử lý text TRƯỚC KHI vào tokenizer, tối ưu cho độ chính xác TTS tiếng Việt.

    Mỗi bước đều có giải thích lý do kỹ thuật ở comment bên dưới.
    Thứ tự bước RẤT QUAN TRỌNG — không nên đảo.

    ⚠️ Hàm này THAY THẾ hoàn toàn addConfigText().
       Nó đã bao gồm logic " . text . " ở Bước 6.

    Gọi từ punc_norm():
        clearText() chạy trước → casefold + xóa dấu câu → rồi mới vào đây.
        Nhưng hàm này được thiết kế để chạy ĐỘC LẬP cũng được (có fallback).

    Args:
        text: văn bản đầu vào
        Tất cả flag khác: bật/tắt từng bước để debug

    Returns:
        str: text đã tối ưu, sẵn sàng đưa vào MTLTokenizer.text_to_tokens()
    """

    if not text or not text.strip():
        return text

    original = text
    step = text

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 0: Unicode NFC Normalization
    # ══════════════════════════════════════════════════════════════════════
    # ĐÂY LÀ BƯỚC QUAN TRỌNG NHẤT.
    #
    # Tiếng Việt Unicode có 2 dạng:
    #   NFC (Composed):   ắ = U+1EAF (1 codepoint, 1 token)
    #   NFD (Decomposed): ắ = a + ̆ + ́  (3 codepoints, 3 tokens!)
    #
    # Tokenizer vocab lưu ký tự dạng NFC precomposed (à=393, á=394, ...)
    # Nhưng tokenizer KHÔNG CÓ normalizer (tokenizer.json: "normalizer": null)
    # → Nếu input là NFD, mỗi combining diacritical mark thành TOKEN RIÊNG:
    #   "ắ" NFD → token [a] + token [̆] + token [́] = 3 tokens thay vì 1
    #
    # Hệ quả: cùng 1 câu, NFD có thể tạo ra GẤP ĐÔI số token so với NFC
    # → T3 nhận dãy token dài hơn cần → dễ "trượt" distribution → sinh noise ở đuôi
    #
    # TÓM LẠI: NFC normalize GIẢM 30-50% token count cho tiếng Việt
    # và là cách HIỆU QUẢ NHẤT để giảm hiện tượng noise đuôi câu dài.
    if unicode_nfc:
        step = unicodedata.normalize('NFC', step)
        if debug:
            delta = len(original) - len(step)
            print(f"  [BƯỚC 0] NFC normalize: {len(original)} → {len(step)} chars (Δ={delta})")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 1: Xóa ký tự invisible / zero-width
    # ══════════════════════════════════════════════════════════════════════
    # Tokenizer CÓ token cho U+200B (ID 2049), U+200C (2050), U+200D (2051)
    # Những ký tự này KHÔNG phát âm nhưng CHIẾM 1 token slot mỗi cái.
    # Text copy từ web/Word thường chứa rất nhiều ký tự loại này.
    # → Xóa hết để tiết kiệm token budget.
    #
    # Cũng thay non-breaking space (U+00A0, ID 329) → space thường
    # vì tokenizer xử lý space qua [SPACE] token (ID 2).
    if remove_invisible:
        before_len = len(step)
        step = step.replace('\u00a0', ' ')  # NBSP → regular space
        step = _INVISIBLE_RE.sub('', step)
        if debug and len(step) != before_len:
            print(f"  [BƯỚC 1] Xóa invisible: {before_len} → {len(step)} chars")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 2: Chuẩn hoá dấu câu
    # ══════════════════════════════════════════════════════════════════════
    # T3 được train chủ yếu với narrative text. Dấu "." (ID 9) là sentence
    # boundary phổ biến nhất → model "tin tưởng" nhất rằng câu đã kết thúc.
    #
    # Dấu "!" (ID 3) và "?" (ID 13) trigger T3 sinh âm điệu lên/xuống
    # vì trong training data chúng đi kèm câu hỏi/cảm thán → prosody change.
    # Khi TTS tiếng Việt cần giọng ổn định → đổi hết về "." hoặc ",".
    #
    # Dấu "," (ID 7) được model hiểu là "pause ngắn, câu chưa hết"
    # → dùng cho ; : — (pause trung bình, không phải dừng hẳn).
    #
    # Đa dấu liên tiếp (... !! ???) → 1 dấu duy nhất:
    #   T3 thấy 3 dấu chấm → 3 tokens [.][.][.] → mỗi token trigger pause
    #   → tổng cộng 3 lần pause → audio bị khoảng lặng dài bất thường.
    if normalize_punctuation:
        step = step.translate(_PUNCT_TO_PERIOD)
        step = re.sub(r'\.{2,}', '.', step)     # .. hoặc ... → .
        step = re.sub(r',{2,}', ',', step)       # ,, hoặc ,,, → ,
        step = re.sub(r'[\.,-]{2,}', '.', step)  # hỗn hợp .,- liên tiếp → .
        if debug:
            print(f"  [BƯỚC 2] Normalize dấu câu: '{step[:80]}...'")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 3: Lọc ký tự ngoài vocab
    # ══════════════════════════════════════════════════════════════════════
    # Vocab có 2549 entries, bao gồm rất nhiều ký tự "rác" (Hebrew, Arabic,
    # CJK, IPA phonetic...) mà model KHÔNG được train phát âm.
    # Ký tự ngoài vocab → [UNK] (ID 1) → T3 nhận token "trống" → noise.
    #
    # Thay vì whitelist (dễ thiếu), ta dùng chính vocab từ tokenizer JSON.
    # Nhưng ta CHỈ cho phép ký tự thuộc các nhóm "phát âm được":
    #   - Chữ Latin (a-z, A-Z, tiếng Việt có dấu)
    #   - Số (0-9)
    #   - Dấu câu cơ bản (. , - ' " space)
    # Tất cả ký tự khác (dù CÓ trong vocab) → thay bằng space.
    if filter_unk_chars:
        cleaned = []
        for ch in step:
            if ch == ' ':
                cleaned.append(ch)
            elif ch in _ALLOWED_CHARS:
                # Chỉ giữ lại nếu ký tự thuộc nhóm "phát âm được"
                cat = unicodedata.category(ch)
                # Lu/Ll = letter; Nd = digit; Po/Pd = punctuation; Mn = combining mark (dấu)
                if cat.startswith(('L', 'N', 'P', 'M')):
                    cleaned.append(ch)
                else:
                    # Ký tự control, symbol, format → thay space
                    cleaned.append(' ')
            else:
                # Ngoài vocab hoàn toàn → space
                cleaned.append(' ')
        step = ''.join(cleaned)
        if debug:
            print(f"  [BƯỚC 3] Filter UNK chars: '{step[:80]}...'")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 4: Chuẩn hoá khoảng trắng
    # ══════════════════════════════════════════════════════════════════════
    # Tokenizer dùng pre_tokenizer: Whitespace → tách theo space TRƯỚC khi BPE.
    # 2 space liên tiếp tạo ra empty string giữa 2 split → encode thành [SPACE]
    # mỗi space thừa = 1 token thừa = lãng phí token budget.
    # Cũng strip đầu/cuối vì boundary pause sẽ thêm " . " riêng.
    if normalize_whitespace:
        step = ' '.join(step.split())  # collapse multiple spaces
        step = step.strip()
        step = step.strip('., ')  # xóa dấu ., thừa ở đầu cuối
        if debug:
            print(f"  [BƯỚC 4] Normalize whitespace: '{step[:80]}...'")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 5: Tách từ quá dài
    # ══════════════════════════════════════════════════════════════════════
    # Tiếng Việt đơn âm tiết: 1 từ = 1-7 ký tự (kể cả dấu).
    # Từ > 12 ký tự thường là:
    #   - URL bị paste nhầm
    #   - Mã sản phẩm (SKU12345XYZ)
    #   - Tên riêng nước ngoài dài không space (Schwarzenegger)
    #
    # BPE tokenizer bẻ từ dài thành nhiều sub-token rất hiếm
    # → T3 không quen pattern → dễ sinh sai, đặc biệt ở cuối câu.
    # → Bẻ ở giữa để tạo 2 phần ngắn hơn, dễ xử lý hơn.
    if split_long_words and step.strip():
        words = step.split()
        new_words = []
        for w in words:
            if len(w) > long_word_threshold:
                # Bẻ ở giữa. Ưu tiên bẻ tại boundary nguyên âm nếu có.
                mid = _find_split_point(w)
                part1 = w[:mid]
                part2 = w[mid:]
                new_words.append(part1)
                new_words.append(part2)
                if debug:
                    print(f"  [BƯỚC 5] Split '{w}' → '{part1}' + '{part2}'")
            else:
                new_words.append(w)
        step = ' '.join(new_words)

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 6: Thêm boundary pause " . text . "
    # ══════════════════════════════════════════════════════════════════════
    # Giống addConfigText() nhưng có thêm safeguard.
    #
    # T3 học từ dataset audio thật: mỗi câu bắt đầu bằng khoảng lặng nhỏ
    # (breath/silence) và kết thúc bằng pause tự nhiên trước khi dừng.
    #
    # Pattern " . text . " tạo ra token sequence:
    #   [SPACE] [.] [SPACE] text [SPACE] [.] [SPACE]
    #   → T3 nhận ra "đây là 1 câu hoàn chỉnh" → sinh audio có đầu/cuối rõ ràng
    #   → GIẢM hiện tượng audio bị cắt cụt hoặc trailing noise
    #
    # Tại sao " . " mà không phải ".":
    #   - pre_tokenizer: Whitespace → "." không có space sẽ DÍNH vào từ trước/sau
    #   - VD: "xin." → tokenize thành ["xin."] → BPE bẻ thành [x][i][n][.]
    #     nhưng " . " → tokenize thành [..., ".", ...] → [.] là token riêng
    #   - Dấu "." là token riêng (ID 9) → T3 hiểu rõ nhất là sentence boundary
    if add_boundary_pause:
        step = step.strip()
        if step:
            step = " . " + step + " . "
        if debug:
            print(f"  [BƯỚC 6] Boundary pause: '{step[:80]}...'")

    if debug:
        # So sánh token count (estimate: số ký tự sau NFC ÷ ~2 cho tiếng Việt)
        orig_chars = len(original)
        final_chars = len(step)
        print(f"  [TỔNG KẾT] original={orig_chars} chars → final={final_chars} chars")

    return step


def _find_split_point(word: str) -> int:
    """
    Tìm điểm bẻ tốt nhất cho từ dài.

    Ưu tiên bẻ tại boundary giữa phụ âm-nguyên âm (dễ phát âm hơn)
    thay vì bẻ giữa 2 nguyên âm hoặc 2 phụ âm.

    Fallback: bẻ ở giữa từ.
    """
    mid = len(word) // 2
    vowels = set('aeiouyàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ')

    # Tìm điểm bẻ consonant→vowel gần mid nhất (trong khoảng ±3)
    best = mid
    best_dist = 999
    for i in range(max(2, mid - 3), min(len(word) - 2, mid + 4)):
        if word[i - 1].lower() not in vowels and word[i].lower() in vowels:
            dist = abs(i - mid)
            if dist < best_dist:
                best = i
                best_dist = dist

    return best


# ════════════════════════════════════════════════════════════════════════════
# TIỆN ÍCH: Hàm debug để kiểm tra text sẽ tạo bao nhiêu token
# ════════════════════════════════════════════════════════════════════════════

def debug_token_analysis(text: str, tokenizer_path: Optional[str] = None):
    """
    Phân tích chi tiết cách tokenizer xử lý text.
    In ra từng token và ID để debug lỗi phát âm.

    Dùng trong console:
        from tts_helper.tts_precision import debug_token_analysis
        debug_token_analysis("xin chào các bạn")
    """
    try:
        from tokenizers import Tokenizer

        if tokenizer_path is None:
            _this_dir = Path(__file__).parent
            tokenizer_path = str(
                _this_dir.parent / "modelViterboxLocal" / "tokenizer_vi_expanded.json"
            )

        tok = Tokenizer.from_file(tokenizer_path)

        print("=" * 70)
        print(f"INPUT: '{text}'")
        print("=" * 70)

        # Phân tích TRƯỚC precision processing
        text_raw = text.replace(' ', '[SPACE]')
        enc_raw = tok.encode(text_raw)
        print(f"\n📌 TRƯỚC config_token_for_precision:")
        print(f"   Token count: {len(enc_raw.ids)}")
        print(f"   Tokens: {enc_raw.tokens}")
        print(f"   IDs:    {enc_raw.ids}")

        # Phân tích SAU precision processing
        processed = config_token_for_precision(text, debug=True)
        text_proc = processed.replace(' ', '[SPACE]')
        enc_proc = tok.encode(text_proc)
        print(f"\n📌 SAU config_token_for_precision:")
        print(f"   Token count: {len(enc_proc.ids)}")
        print(f"   Tokens: {enc_proc.tokens}")
        print(f"   IDs:    {enc_proc.ids}")

        # So sánh
        delta = len(enc_raw.ids) - len(enc_proc.ids)
        pct = (delta / max(len(enc_raw.ids), 1)) * 100
        print(f"\n📊 Giảm: {delta} tokens ({pct:.1f}%)")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Lỗi khi phân tích: {e}")
        import traceback
        traceback.print_exc()
