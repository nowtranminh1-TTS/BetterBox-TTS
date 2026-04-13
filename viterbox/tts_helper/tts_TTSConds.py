"""
TTSConds — Container cho conditioning tensors dùng trong TTS generation.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Union

from ..models.t3.modules.cond_enc import T3Cond


@dataclass
class TTSConds:
    """Conditioning tensors for TTS generation"""
    t3: Union['T3Cond', dict]   # T3 conditioning (T3Cond object or dict)
    s3: dict                     # S3Gen conditioning dict
    ref_wav: Optional[torch.Tensor] = None

    def save(self, path):
        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            elif isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            elif hasattr(x, '__dict__'):
                return {k: to_cpu(v) for k, v in vars(x).items()}
            return x

        torch.save({
            't3': to_cpu(self.t3),
            'gen': to_cpu(self.s3),
        }, path)

    @classmethod
    def load(cls, path, device):
        def to_device(x, dev):
            if isinstance(x, torch.Tensor):
                return x.to(dev)
            elif isinstance(x, dict):
                return {k: to_device(v, dev) for k, v in x.items()}
            return x

        data = torch.load(path, map_location='cpu', weights_only=False)

        # Handle both old format (t3, s3) and new format (t3, gen)
        t3_data = data.get('t3', {})
        s3_data = data.get('gen', data.get('s3', {}))
        ref_wav  = data.get('ref_wav', None)

        # Convert t3_data dict to T3Cond object
        if isinstance(t3_data, dict) and 'speaker_emb' in t3_data:
            t3_cond = T3Cond(
                speaker_emb=to_device(t3_data['speaker_emb'], device),
                cond_prompt_speech_tokens=to_device(t3_data.get('cond_prompt_speech_tokens'), device),
                cond_prompt_speech_emb=(
                    to_device(t3_data['cond_prompt_speech_emb'], device)
                    if t3_data.get('cond_prompt_speech_emb') is not None else None
                ),
                clap_emb=(
                    to_device(t3_data['clap_emb'], device)
                    if t3_data.get('clap_emb') is not None else None
                ),
                emotion_adv=(
                    to_device(t3_data['emotion_adv'], device)
                    if t3_data.get('emotion_adv') is not None else None
                ),
            )
        else:
            t3_cond = to_device(t3_data, device)

        return cls(
            t3=t3_cond,
            s3=to_device(s3_data, device),
            ref_wav=to_device(ref_wav, device) if ref_wav is not None else None,
        )
