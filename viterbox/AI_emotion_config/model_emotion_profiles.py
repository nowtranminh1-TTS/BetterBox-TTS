"""
Model Emotion Profiles — Cấu hình cảm xúc dựa trên tham số thực sự của model AI (T3/Viterbox)

Mỗi profile điều chỉnh trực tiếp các tham số generation của model:
  • exaggeration (emotion_adv) — Độ biểu cảm prosody
      0.0 = cảm xúc mạnh, sắc nét ở cuối câu
      2.0 = âm đuôi mượt, trung tính hơn
  • cfg_weight — Classifier-Free Guidance, kiểm soát độ bám sát text
      Cao hơn = đọc đúng từ, kiểm soát chặt
      Thấp hơn = tự do hơn, biểu cảm hơn
  • temperature — Độ ngẫu nhiên khi sampling speech token
      Cao hơn = biến đổi prosody nhiều hơn, sống động
      Thấp hơn = ổn định, đều đều
  • top_p — Ngưỡng nucleus sampling
      Cao hơn = đa dạng token hơn
      Thấp hơn = chọn token an toàn

Lưu ý:
  Model được train với exaggeration trong khoảng [0, 2].
  Giá trị > 2 có thể hoạt động nhưng nằm ngoài phân phối training
  và có nguy cơ gây artifact.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class ModelEmotionProfile:
    """Một profile cảm xúc dựa trên tham số generation thực sự của model."""
    name: str                       # key duy nhất
    display_name: str               # tên hiển thị trong UI
    description: str                # mô tả ngắn
    exaggeration: Optional[float]   # None ⟹ dùng giá trị slider hiện tại
    cfg_weight: Optional[float]     # None ⟹ dùng giá trị slider hiện tại
    temperature: Optional[float]    # None ⟹ dùng giá trị slider hiện tại
    top_p: Optional[float]          # None ⟹ dùng giá trị slider hiện tại
    rep_pen: Optional[float]        # None ⟹ dùng giá trị slider hiện tại


# ──────────────────────────────────────────────────────────────────────
#  Registry
# ──────────────────────────────────────────────────────────────────────

_PROFILES: Dict[str, ModelEmotionProfile] = {}


def _register(p: ModelEmotionProfile):
    _PROFILES[p.name] = p
    return p

# ── Custom ──────────────────────────────────────────────────────────
_register(ModelEmotionProfile(
    name="AI-custom",
    display_name="🛠️ Custom - user tự chỉnh trên slider",
    description="Mặc định — dùng giá trị slider, không override tham số",
    exaggeration=None,      # dùng slider
    cfg_weight=None,      # dùng slider
    temperature=None,      # dùng slider
    top_p=None,      # dùng slider
    rep_pen=None,      # dùng slider
))

# ── Precision ──────────────────────────────────────────────────────────
_register(ModelEmotionProfile(
    name="AI-precision",
    display_name="🎯 Precision (default) - siêu chính xác",
    description="tối ưu để đọc chuẩn nhất, override tham số",
    exaggeration=2.0,     
    cfg_weight=2.0,      
    temperature=0.1,      
    top_p=0.1,  
    rep_pen=1.0,    
))


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def get_model_emotion_profile(name: str) -> ModelEmotionProfile:
    """Lấy profile theo tên.  Fallback về 'AI-custom' nếu không tồn tại."""
    return _PROFILES.get(name, _PROFILES["AI-custom"])


def list_model_emotion_profiles() -> List[str]:
    """Trả về danh sách tên tất cả profile."""
    return list(_PROFILES.keys())


def get_model_emotion_choices() -> List[Tuple[str, str]]:
    """Trả về list (display_name, key) cho Gradio Dropdown."""
    return [(p.display_name, p.name) for p in _PROFILES.values()]


def get_model_emotion_description(name: str) -> str:
    """Trả về mô tả ngắn của profile, kèm thông số."""
    p = _PROFILES.get(name, _PROFILES["AI-custom"])
    parts = [p.description]
    if p.exaggeration is not None:
        parts.append(f"exag={p.exaggeration}")
    parts.append(f"cfg={p.cfg_weight}, temp={p.temperature}, top_p={p.top_p}")
    return " | ".join(parts)
