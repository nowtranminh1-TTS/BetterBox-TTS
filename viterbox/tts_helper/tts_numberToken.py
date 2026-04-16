# viterbox/tts_helper/tts_numberToken.py

import unicodedata

from general.general_tool_audio import(
    clearText, 
    normalize_text
)

def get_list_word(word: str) -> list:
    # Chuẩn hóa về dạng 'dựng sẵn' để các chữ có dấu không bị tách rời
    word = unicodedata.normalize('NFC', word)
    return list(word)

def getNumberTokenText(content: str, input_token_count: int) -> int:
    # Xử lý text - BUỘC PHẢI CÓ ĐỂ CHUẨN HÓA
    getContent = clearText(content)
    getContent = normalize_text(getContent)

    bunchOfText = getContent.split()

    print(f"\n💎 Số token: {input_token_count}, 🔠chữ trong câu: {len(bunchOfText)}, {bunchOfText}")

    n = len(bunchOfText)

    if n == 1:
        # sẽ sử dụng cái này nhiều nhất cho advance mode TTS
        # vì cách hoạt động của hàm inference là tách từng chữ ra inference rồi ráp kết quả TTS lại 
        getNumber = number_token_for_single_word(n, getContent, input_token_count)
        return getNumber
    elif n == 2:
        buffer = input_token_count * 0.15  # tăng thêm 15% để tránh bị thiếu
        return min(int(input_token_count + buffer), 1000)
    elif n == 3:
        buffer = input_token_count * 0.15  # tăng thêm 15% để tránh bị thiếu
        return min(int(input_token_count + buffer), 1000)
    elif n == 4:
        buffer = input_token_count * 0.15  # tăng thêm 15% để tránh bị thiếu
        return min(int(input_token_count + buffer), 1000)
    
    elif n > 4 and n <= 17:
        buffer = input_token_count * 0.40  # tăng thêm 40% để tránh bị thiếu
        return min(int(input_token_count + buffer), 1000)
    else:
        buffer = input_token_count * 0.40  # tăng thêm 40% để tránh bị thiếu
        return min(int(input_token_count + buffer), 1000)
    
    
def number_token_for_single_word(number_of_words: int, 
                                text: str, 
                                input_token_count: int) -> int:
    # Xử lý text - BUỘC PHẢI CÓ ĐỂ CHUẨN HÓA
    text = clearText(text)
    text = normalize_text(text)
    text = text.casefold()  # đảm bảo chữ thường hết

    getNormal = min(int(input_token_count), 1000) # lấy full
    getSpecial = min(int(input_token_count * 0.85), 1000) # chỉ lấy 85% token

    if number_of_words > 1: 
        return getNormal
    
    # ta mạc định text là chỉ có 1 chữ 
    listWord = get_list_word(text) # EX: 'chào' -> ['c', 'h', 'à', 'o']
    listWord_lower = {w.casefold() for w in listWord}  # Ép toàn bộ danh sách về lower

    # Tập hợp các ký tự đặc biệt cần kiểm tra
    special_words = {'ạ', 'ậ', 'ặ', 'ẹ', 'ệ', 'ị', 'ọ', 'ộ', 'ợ', 'ụ', 'ự', 'ỵ'}
    special_words_lower = {w.casefold() for w in special_words}  # Ép toàn bộ danh sách về lower

    is_contant_special_word = any(char in special_words_lower for char in listWord_lower)

    

    if is_contant_special_word:
        print(f"1️⃣ MỘT chữ SPECIAL: {text}, 📝 và các từ của chữ đó: {listWord_lower} \n")
        return getSpecial
    else:
        print(f"1️⃣ MỘT chữ NORMAL: {text}, 📝 và các từ của chữ đó: {listWord_lower} \n")
        return getNormal