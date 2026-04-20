"""ASR Chunkformer module for OmniVoice.

This module provides Vietnamese ASR functionality using Chunkformer model.
"""

import logging
import os
import sys
import tempfile
from typing import Optional, Union

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class ASRChunkformer:
    """Chunkformer ASR wrapper for Vietnamese speech recognition.
    
    This class wraps the Chunkformer model for use in OmniVoice voice cloning.
    The model is loaded from local path and optimized for Vietnamese speech.
    """
    
    _model: "ChunkFormerModel"  # Type annotation for IDE navigation
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize ASR Chunkformer.
        
        Args:
            device: Device to load model on ("cpu" or "cuda").
                   If None, auto-detect GPU if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None
        self._project_root = self._get_project_root()
    
    def _get_project_root(self) -> str:
        """Get project root directory (OmniVoice/)."""
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # omnivoice/models/ -> go up 2 levels to OmniVoice/
        return os.path.dirname(os.path.dirname(current_file_dir))
    
    def _ensure_chunkformer_in_path(self):
        """Add chunkformer lib to sys.path if not already present."""
        chunkformer_path = os.path.join(self._project_root, "chunkformer")
        if chunkformer_path not in sys.path:
            sys.path.insert(0, chunkformer_path)
    
    def _patch_pydub_for_torchaudio(self):
        """Monkey-patch pydub to use torchaudio instead of ffmpeg.
        
        This eliminates the need for ffmpeg system dependency.
        """
        try:
            import pydub.audio_segment
            import pydub.utils
            from pydub.audio_segment import AudioSegment
            
            # Store original from_file
            if not hasattr(AudioSegment, '_original_from_file'):
                AudioSegment._original_from_file = AudioSegment.from_file
            
            def _torchaudio_from_file(file, format=None, **kwargs):
                """Replacement that uses torchaudio instead of ffmpeg."""
                import torchaudio
                import io
                
                # Handle file path or file-like object
                if hasattr(file, 'read'):
                    # It's a file-like object
                    file.seek(0)
                    data = file.read()
                    waveform, sample_rate = torchaudio.load(io.BytesIO(data))
                else:
                    # It's a file path
                    waveform, sample_rate = torchaudio.load(file)
                
                # Convert to pydub AudioSegment format
                # pydub uses raw audio data with specific parameters
                samples = waveform.numpy().T  # (channels, samples) -> (samples, channels)
                if samples.ndim == 1:
                    samples = samples.reshape(-1, 1)
                
                # Convert to bytes
                sample_width = 2  # 16-bit
                samples = (samples * (2**15 - 1)).astype(np.int16)
                raw_data = samples.tobytes()
                
                # Create AudioSegment
                segment = AudioSegment(
                    data=raw_data,
                    sample_width=sample_width,
                    frame_rate=sample_rate,
                    channels=samples.shape[1] if samples.ndim > 1 else 1
                )
                return segment
            
            # Replace the method
            AudioSegment.from_file = staticmethod(_torchaudio_from_file)
            
            # Also patch mediainfo_json if needed
            original_mediainfo = pydub.utils.mediainfo_json
            def _patched_mediainfo_json(filepath, read_ahead_limit=-1):
                """Minimal mediainfo that returns basic audio info."""
                import torchaudio
                info = torchaudio.info(filepath)
                return {
                    'streams': [{
                        'codec_type': 'audio',
                        'sample_rate': info.sample_rate,
                        'channels': info.num_channels,
                        'duration': info.num_frames / info.sample_rate if info.sample_rate > 0 else 0,
                    }],
                    'format': {'format_name': 'wav'}
                }
            pydub.utils.mediainfo_json = _patched_mediainfo_json
            
            print(f"🔧[ASR] Đã patch pydub để dùng torchaudio (không cần ffmpeg)")
            
        except Exception as e:
            print(f"⚠️[ASR] Không thể patch pydub: {e}, sẽ dùng ffmpeg nếu có")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load Chunkformer ASR model.
        
        Args:
            model_path: Path to model directory. If None, uses default model
                       in project root: model_ASR_chunkformer_local
        """
        self._ensure_chunkformer_in_path()
        
        # Patch pydub to use torchaudio instead of ffmpeg
        self._patch_pydub_for_torchaudio()
        
        if model_path is None:
            model_path = os.path.join(self._project_root, "model_ASR_chunkformer_local")
        
        from chunkformer import ChunkFormerModel
        
        logger.info("Loading Chunkformer ASR model from %s ...", model_path)
        print(f"🔄 Đang load Chunkformer ASR model từ: {model_path}")
        
        try:
            self._model = ChunkFormerModel.from_pretrained(model_path)
            self._model.to(self.device)
            self._model.eval()
            print(f"✅ Chunkformer ASR model load thành công trên {self.device}\n")
            logger.info("Chunkformer ASR model loaded on %s.", self.device)
        except Exception as e:
            print(f"❌ Chunkformer ASR model load thất bại: {e}")
            raise
    
    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, tuple],
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio using the loaded Chunkformer ASR model.
        
        Args:
            audio: File path or (waveform, sample_rate) tuple.
                Waveform can be numpy array or torch.Tensor of shape
                (1, T) or (T,).
            language: Language code (kept for API compatibility, ignored
                     since Chunkformer Vietnamese model is monolingual).
        
        Returns:
            Transcribed text.
        """
        if self._model is None:
            print("❌ Lỗi: ASR model chưa được load. Gọi load_model() trước.")
            raise RuntimeError(
                "ASR model is not loaded. Call load_model() first."
            )
        
        print(f"🎯 Bắt đầu transcribe audio...")
        
        # Handle audio input
        if isinstance(audio, str):
            # File path - use directly
            audio_path = audio
            print(f"📁 Input là file path: {audio_path}")
        else:
            # Tuple of (waveform, sample_rate) - save to temp file
            waveform, sr = audio
            print(f"🔊 Input là waveform tensor, sample_rate={sr}")
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            waveform = np.squeeze(waveform)  # (1, T) or (T,) → (T,)
            
            # Save to temp file - ensure file is fully written and closed
            import tempfile
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            try:
                os.close(fd)  # Close file descriptor immediately
                waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
                torchaudio.save(audio_path, waveform_tensor, sr)
                print(f"💾 [ASR] Đã lưu temp file: {audio_path}")
                
                result = self._transcribe_file(audio_path)
            finally:
                # Cleanup with retry for Windows file locking
                if os.path.exists(audio_path):
                    import time
                    for _ in range(3):  # Try 3 times
                        try:
                            os.remove(audio_path)
                            print(f"🗑️[ASR] Đã xóa temp file")
                            break
                        except PermissionError:
                            time.sleep(0.1)  # Wait a bit for file to be released
                    else:
                        print(f"⚠️ [ASR] Không thể xóa temp file: {audio_path}")
            
            return result
        
        # For file path input
        result = self._transcribe_file(audio_path)
        print(f"📝[ASR] Transcribe thành công: '{result[:50]}{'...' if len(result) > 50 else ''}'")
        return result
    
    def _transcribe_file(self, audio_path: str) -> str:
        """Internal method to transcribe audio file.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Transcribed text string.
        """
        print(f"🚀[ASR] Đang transcribe file: {audio_path}")
        try:
            result = self._model.endless_decode(
                audio_path=audio_path,
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=14400,  # 4 hours max
                return_timestamps=False,  # Just get text
                max_silence_duration=0.5,
            )
            print(f"✅[ASR] endless_decode hoàn thành")
            return result.strip() if isinstance(result, str) else str(result).strip()
        except Exception as e:
            print(f"❌[ASR] Lỗi khi transcribe: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
