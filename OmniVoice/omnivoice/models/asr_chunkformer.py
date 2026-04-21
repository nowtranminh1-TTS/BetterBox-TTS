"""ASR Chunkformer module for OmniVoice.

This module provides Vietnamese ASR functionality using Chunkformer model.
"""

import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Optional, Union, Generator

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

    def _patch_pydub(self):
        """Monkey-patch pydub to use torchaudio instead of ffmpeg.
        
        This eliminates the need for ffmpeg system dependency.
        Must be called before importing ChunkFormerModel.
        """
        try:
            import pydub.audio_segment
            import pydub.utils
            from pydub.audio_segment import AudioSegment
            
            # Avoid double-patch
            if hasattr(AudioSegment, '_torchaudio_patched'):
                return
            
            # Store original
            AudioSegment._original_from_file = AudioSegment.from_file
            
            @staticmethod
            def _torchaudio_from_file(file, format=None, **kwargs):
                """Replacement that uses torchaudio instead of ffmpeg."""
                import io
                
                # Handle file path or file-like object
                if hasattr(file, 'read'):
                    file.seek(0)
                    data = file.read()
                    waveform, sample_rate = torchaudio.load(io.BytesIO(data))
                else:
                    waveform, sample_rate = torchaudio.load(file)
                
                # Convert to pydub AudioSegment format
                samples = waveform.numpy().T  # (channels, samples) -> (samples, channels)
                if samples.ndim == 1:
                    samples = samples.reshape(-1, 1)
                
                # Convert to bytes (16-bit)
                samples = (samples * (2**15 - 1)).astype(np.int16)
                raw_data = samples.tobytes()
                
                # Create AudioSegment
                segment = AudioSegment(
                    data=raw_data,
                    sample_width=2,  # 16-bit
                    frame_rate=sample_rate,
                    channels=samples.shape[1] if samples.ndim > 1 else 1
                )
                return segment
            
            AudioSegment.from_file = _torchaudio_from_file
            AudioSegment._torchaudio_patched = True
            
            # Patch mediainfo_json
            def _patched_mediainfo_json(filepath, read_ahead_limit=-1):
                """Minimal mediainfo that returns basic audio info."""
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
            
            pydub.utils._original_mediainfo_json = pydub.utils.mediainfo_json
            pydub.utils.mediainfo_json = _patched_mediainfo_json
            
            print(f"🔧[ASR] Đã patch pydub để dùng torchaudio (không cần ffmpeg)")
            
        except Exception as e:
            print(f"⚠️[ASR] Không thể patch pydub: {e}")

    def load_model(self, model_path: Optional[str] = None):
        """Load Chunkformer ASR model.
        
        Args:
            model_path: Path to model directory. If None, uses default model
                       in project root: model_ASR_chunkformer_local
        """
        self._ensure_chunkformer_in_path()
        
        # Patch pydub to use torchaudio instead of ffmpeg (must be before importing ChunkFormerModel)
        self._patch_pydub()
        
        if model_path is None:
            model_path = os.path.join(self._project_root, "model_ASR_chunkformer_local")
        
        from chunkformer import ChunkFormerModel
        
        logger.info("[ASR]Loading Chunkformer ASR model from %s ...", model_path)
        print(f"🔄 [ASR]Đang load Chunkformer ASR model từ: {model_path}")
        
        try:
            self._model = ChunkFormerModel.from_pretrained(model_path)
            self._model.to(self.device)
            self._model.eval()
            print(f"✅ Chunkformer ASR model load thành công trên {self.device}\n")
            logger.info("Chunkformer ASR model loaded on %s.", self.device)
        except Exception as e:
            print(f"❌ Chunkformer ASR model load thất bại: {e}")
            raise

    @contextmanager
    def _temp_audio_file(self, suffix: str = ".wav") -> Generator[str, None, None]:
        """Context manager for temp audio file with automatic cleanup."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            yield path
        finally:
            self._cleanup_file(path)

    def _cleanup_file(self, path: str) -> None:
        """Remove file with retry for Windows file locking."""
        if os.path.exists(path):
            for _ in range(3):
                try:
                    os.remove(path)
                    print(f"🗑️[ASR] Đã xóa temp file")
                    return
                except PermissionError:
                    time.sleep(0.1)
            print(f"⚠️ [ASR] Không thể xóa temp file: {path}")

    @torch.inference_mode()
    def transcribe(self, audio: Union[str, tuple], language: Optional[str] = "vi") -> str:
        """Transcribe audio using the loaded Chunkformer ASR model.
        
        Args:
            audio: File path or (waveform, sample_rate) tuple.
            language: Ignored (Chunkformer Vietnamese model is monolingual).
        
        Returns:
            Transcribed text.
        """
        if self._model is None:
            raise RuntimeError("ASR model is not loaded. Call load_model() first.")
        print(f"🎯 Bắt đầu transcribe audio...")
        with self._temp_audio_file() as temp_path:
            self._prepare_audio(audio, temp_path)
            result = self._transcribe_file(temp_path)
            print(f"📝[ASR] Transcribe thành công: '{result[:50]}{'...' if len(result) > 50 else ''}'")
            return result

    def _prepare_audio(self, audio: Union[str, tuple], output_path: str, target_sr: int = 16000) -> None:
        """Prepare audio file at output_path from various input types."""
        if isinstance(audio, str):
            print(f"📁 [Chunkformer ASR] Input là file path: {audio}")
            waveform, orig_sr = torchaudio.load(audio)
        else:
            waveform, orig_sr = audio
            print(f"🔊 [Chunkformer ASR] Input là waveform tensor, sample_rate={orig_sr}")
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu()
            else:
                waveform = torch.from_numpy(np.array(waveform))
            # Ensure shape is (channels, samples)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)

        torchaudio.save(output_path, waveform, target_sr)
        print(f"💾[ASR] Đã lưu temp file: {output_path}")

    def _transcribe_file(self, audio_path: str) -> str:
        """Internal method to transcribe audio file.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Transcribed text string.
        """
        print(f"🚀[ASR] Đang transcribe file: {audio_path}")
        result = self._model.endless_decode(
            audio_path=audio_path,
            chunk_size=64,
            left_context_size=128,
            right_context_size=128,
            total_batch_duration=14400,
            return_timestamps=False,
            max_silence_duration=0.5,
        )
        return result.strip() if isinstance(result, str) else str(result).strip()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
