"""
EQ emotion configuration package for Viterbox.
"""

from .eq_emotional_profiles import (
    apply_amplitude_envelope,
    get_emotional_audio_profile,
    get_profile_description,
    list_emotional_profiles,
)

__all__ = [
    "apply_amplitude_envelope",
    "get_emotional_audio_profile",
    "get_profile_description",
    "list_emotional_profiles",
]
