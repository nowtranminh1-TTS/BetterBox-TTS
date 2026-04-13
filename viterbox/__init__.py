"""
Viterbox - Vietnamese Text-to-Speech
"""
from .tts import Viterbox
from .tts_helper.tts_TTSConds import TTSConds   # moved to tts_helper/

__version__ = "1.0.0"
__all__ = ["Viterbox", "TTSConds"]
