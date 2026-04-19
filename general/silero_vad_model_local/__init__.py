from importlib.metadata import version
try:
    __version__ = version(__name__)
except:
    pass

from .model import load_silero_vad
from .utils_vad import (get_speech_timestamps,
                        save_audio,
                        read_audio,
                        VADIterator,
                        collect_chunks,
                        drop_chunks)
