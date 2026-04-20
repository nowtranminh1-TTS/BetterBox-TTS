# Type imports for IDE navigation support
# These imports help with Ctrl+Click navigation in IDEs
# The actual runtime imports happen in asr_chunkformer.py with dynamic sys.path

try:
    # Try to import chunkformer for IDE type checking
    # If chunkformer is not in PYTHONPATH, this will fail silently at runtime
    # but the import statement helps IDE with navigation
    import sys
    import os
    
    # Add chunkformer to path for IDE support
    _chunkformer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "chunkformer"
    )
    if _chunkformer_path not in sys.path:
        sys.path.insert(0, _chunkformer_path)
    
    from chunkformer import ChunkFormerModel
    from chunkformer.chunkformer_model import ChunkFormerModel as _ChunkFormerModel
    
except Exception:
    # If import fails (e.g., chunkformer not installed), ignore
    # Runtime imports handle this in asr_chunkformer.py
    pass

# Export ASRChunkformer
from .asr_chunkformer import ASRChunkformer
from .omnivoice import OmniVoice

__all__ = ["ASRChunkformer", "OmniVoice"]