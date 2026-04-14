#!/usr/bin/env python3
"""
Viterbox Emotional Audio Demo - Showcase Emotional Audio Profiles
Demo script để hiển thị các emotional audio profiles với cảm xúc
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path so `import viterbox` works when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from viterbox.tts import Viterbox
from general.EQ_emotion_config.eq_emotional_profiles import (
    list_emotional_profiles,
    get_emotional_audio_profile,
    get_profile_description,
)


def demo_emotional_profiles():
    """Demo tất cả emotional profiles có sẵn"""
    
    print("🎭 Viterbox Emotional Audio Profiles Demo")
    print("=" * 50)
    
    # Liệt kê các profiles
    profiles = list_emotional_profiles()
    print(f"📋 Available emotional profiles: {profiles}")
    print()
    
    # Test text cho demo - với các câu có cảm xúc khác nhau
    test_texts = {
        "happy": "Tôi thật vui vẻ khi được gặp bạn hôm nay! Cuộc sống thật tuyệt vời!",
        "excited": "TUYỆT VỜI! Chúng ta đã làm được rồi! Đây là thành quả của tất cả mọi người!",
        "calm": "Hãy hít thở sâu và thư giãn. Mọi thứ đều ổn cả.",
        "sad": "Tôi nhớ những ngày xưa ấy... Kỷ niệm nào cũng thật ý nghĩa.",
        "dramatic": "Đây là khoảnh khắc quyết định số phận! Lịch sử sẽ ghi nhớ!"
    }
    
    try:
        # Load model với default profile
        tts = Viterbox.from_pretrained("cuda")
        print(f"✅ Model loaded with profile: {tts.get_current_profile()}")
        print()
        
        # Demo từng emotional profile
        for profile in profiles:
            print(f"🎭 Testing emotional profile: {profile}")
            print(f"📝 Description: {get_profile_description(profile)}")
            print("-" * 50)
            
            # Switch to emotional profile
            tts.switch_emotional_profile(profile)
            
            # Get appropriate test text
            test_text = test_texts.get(profile, "Xin chào, đây là demo hệ thống Viterbox với emotional audio.")
            
            print(f"🗣  Generating: {test_text}")
            
            # Generate audio
            audio = tts.generate(test_text)
            
            # Save with profile name
            output_file = f"emotional_{profile}.wav"
            tts.save_audio(audio, output_file)
            
            print(f"✅ Generated: {output_file}")
            print(f"📊 Audio shape: {audio.shape}")
            print(f"🔊 Sample rate: {tts.sr} Hz")
            print(f"🎛️ Current profile: {tts.get_current_profile()}")
            print()
        
        print("🎉 Emotional demo completed! Check the generated WAV files.")
        print("🔍 Compare the emotional quality across different profiles:")
        for profile in profiles:
            desc = get_profile_description(profile)
            print(f"   - {profile}: {desc}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure:")
        print("   - Model files are in viterbox/modelViterboxLocal/")
        print("   - CUDA is available")
        print("   - pedalboard is installed: pip install pedalboard")


def demo_emotional_effects_chain():
    """Demo chi tiết các hiệu ứng trong emotional profiles"""
    
    print("\n🔧 Emotional Audio Effects Chain Analysis")
    print("=" * 50)
    
    try:
        # Get từng profile để inspect
        profiles = list_emotional_profiles()
        
        for profile in profiles:
            print(f"\n🎭 {profile.upper()} PROFILE:")
            print(f"📝 {get_profile_description(profile)}")
            
            board = get_emotional_audio_profile(profile)
            
            print("🔗 Audio Chain:")
            for i, plugin in enumerate(board):
                plugin_name = plugin.__class__.__name__
                print(f"   {i+1}. {plugin_name}")
                
                # Print key parameters
                if hasattr(plugin, '__dict__'):
                    for attr, value in plugin.__dict__.items():
                        if not attr.startswith('_'):
                            print(f"      - {attr}: {value}")
                print()
        
        # Demo processing với audio dummy
        print("🎵 Testing emotional audio processing...")
        dummy_audio = np.random.randn(24000).astype(np.float32) * 0.1  # 1 second at 24kHz
        
        for profile in profiles:
            board = get_emotional_audio_profile(profile)
            processed = board(dummy_audio, 24000)
            print(f"✅ {profile}: {len(dummy_audio)} → {len(processed)} samples")
        
    except Exception as e:
        print(f"❌ Effects analysis failed: {e}")


def demo_profile_switching():
    """Demo switching giữa profiles tại runtime"""
    
    print("\n🔄 Runtime Profile Switching Demo")
    print("=" * 50)
    
    try:
        tts = Viterbox.from_pretrained("cuda")
        
        test_text = "Đây là bài test chuyển đổi profile trong runtime."
        
        print(f"🎛️ Starting with: {tts.get_current_profile()}")
        print(f"🗣  Text: {test_text}")
        print()
        
        profiles = ["no_eq_processing"] + list_emotional_profiles()
        
        for i, profile in enumerate(profiles):
            if profile == "no_eq_processing":
                tts.board = None
                tts.emotional_profile = None
            else:
                tts.switch_emotional_profile(profile)
            
            print(f"🎭 Profile {i+1}: {tts.get_current_profile()}")
            
            # Generate short audio
            audio = tts.generate(test_text[:50])  # Shorter text for demo
            output_file = f"switch_{i+1}_{profile}.wav"
            tts.save_audio(audio, output_file)
            
            print(f"   ✅ Saved: {output_file}")
            print()
        
        print("🎉 Runtime switching demo completed!")
        
    except Exception as e:
        print(f"❌ Runtime switching failed: {e}")


def create_emotional_guide():
    """Tạo guide sử dụng emotional audio profiles"""
    
    guide = """
🎭 VITERBOX EMOTIONAL AUDIO PROFILES GUIDE
==========================================

## 🌟 OVERVIEW
Viterbox Emotional Audio Profiles sử dụng Spotify Pedalboard để tạo ra âm thanh có cảm xúc, đầy biểu cảm và cuốn hút.

## 🎛️ AVAILABLE EMOTIONAL PROFILES

### 🌞 HAPPY - VUI VẺ
- **Mục đích**: Sáng sủa, năng lượng, lạc quan
- **Hiệu ứng chính**: Bright chorus, sparkle delay, airy reverb
- **Use case**: Content vui vẻ, motivational, educational
- **Audio chain**: Highpass(100Hz) → Lowpass(12kHz) → Gain(+2dB) → Chorus → Compressor → Delay → Reverb → Limiter

### 🔥 EXCITED - NĂNG ĐỘNG  
- **Mục đích**: Năng lượng cao, sôi động, mạnh mẽ
- **Hiệu ứng chính**: Chorus+Phaser combo, rhythmic delay, big reverb
- **Use case**: Thể thao, gaming, action content
- **Audio chain**: Highpass(120Hz) → Lowpass(10kHz) → Gain(+4dB) → Mix(Chorus+Phaser) → Compressor → Delay → Reverb → Limiter

### 🌊 CALM - BÌNH YÊN
- **Mục đích**: Dịu dàng, thư giãn, ấm áp
- **Hiệu ứng chính**: Subtle chorus, warm delay, cozy reverb
- **Use case**: Meditation, storytelling, sleep content
- **Audio chain**: Highpass(60Hz) → Lowpass(7kHz) → Gain(-1dB) → Chorus → Compressor → Delay → Reverb → Limiter

### 💧 SAD - BUỒN BÃ
- **Mục đích**: Melancholic, sâu lắng, cảm xúc
- **Hiệu ứng chính**: Melancholic chorus, echoing delay, somber reverb
- **Use case**: Drama, emotional storytelling
- **Audio chain**: Highpass(50Hz) → Lowpass(6kHz) → Gain(-2dB) → Chorus → Compressor → Delay → Reverb → Limiter

### 🎭 DRAMATIC - KỊCH TÍNH
- **Mục đích**: Mạnh mẽ, ấn tượng, điện ảnh
- **Hiệu ứng chính**: Dramatic chorus+phaser, epic delay, cinematic reverb
- **Use case**: Movie trailer, epic content
- **Audio chain**: Highpass(80Hz) → Lowpass(9kHz) → Gain(+1dB) → Mix(Chorus+Phaser) → Compressor → Delay → Reverb → Limiter

## 🚀 USAGE EXAMPLES

### Basic Usage:
```python
# Load với emotional profile
tts = Viterbox.from_pretrained("cuda", emotional_profile="happy")

# Switch emotional profile tại runtime
tts.switch_emotional_profile("excited")

# Switch về audio raw
tts.board = None
tts.emotional_profile = None

# Check current mode
if tts.is_emotional_mode():
    print(f"Using emotional profile: {tts.get_current_profile()}")
```

### Advanced Usage:
```python
# List tất cả emotional profiles
emotional = tts.list_emotional_profiles()
print(emotional)  # ['happy', 'excited', 'calm', 'sad', 'dramatic']

# Get description
desc = tts.get_emotional_profile_description("happy")
print(desc)  # "🌞 VUI VẺ - Sáng sủa, năng lượng, lạc quan"

# Generate với emotional context
tts.switch_emotional_profile("dramatic")
audio = tts.generate("Đây là khoảnh khắc quyết định!")
tts.save_audio(audio, "dramatic_speech.wav")
```

## 🎯 BEST PRACTICES

### Choosing the Right Profile:
- **Happy**: Use for positive, uplifting content
- **Excited**: Use for high-energy, action content  
- **Calm**: Use for relaxing, educational content
- **Sad**: Use for emotional, dramatic storytelling
- **Dramatic**: Use for cinematic, impactful moments

### Performance Tips:
- Emotional profiles use more effects → slightly slower processing
- Switch profiles before generating long content
- Test different profiles with your specific content
- Combine with text content that matches the emotion

### Technical Notes:
- All profiles maintain speech intelligibility
- Built on Spotify Pedalboard professional audio effects
- Optimized for Vietnamese TTS characteristics
- Compatible with all existing Viterbox features

## 🔧 CUSTOMIZATION

You can modify emotional profiles in `general/EQ_emotion_config/eq_emotional_profiles.py`:
- Adjust effect parameters for your preference
- Add new emotional profiles
- Combine different effects chains

## 📊 COMPARISON

| Profile | Energy | Brightness | Reverb | Best For |
|---------|--------|------------|--------|----------|
| Happy | Medium | High | Light | Positive content |
| Excited | High | High | Heavy | Action/gaming |
| Calm | Low | Low | Minimal | Meditation |
| Sad | Low | Dark | Medium | Drama |
| Dramatic | High | Medium | Heavy | Cinema |

💡 **Tip**: Start with "happy" for general use, then experiment with others!
"""
    
    with open("EMOTIONAL_AUDIO_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("📖 Created EMOTIONAL_AUDIO_GUIDE.md with comprehensive usage instructions")


if __name__ == "__main__":
    print("🎭 Viterbox Emotional Audio Demo Suite")
    print("=====================================")
    
    # Check requirements
    try:
        import pedalboard
        print("✅ Pedalboard installed")
    except ImportError:
        print("❌ Pedalboard not found. Install with: pip install pedalboard")
        sys.exit(1)
    
    # Run demos
    demo_emotional_profiles()
    demo_emotional_effects_chain()
    demo_profile_switching()
    create_emotional_guide()
    
    print("\n🎉 Emotional audio demo suite completed!")
    print("📁 Check generated files:")
    print("   - emotional_*.wav (emotional audio samples)")
    print("   - switch_*.wav (runtime switching demo)")
    print("   - EMOTIONAL_AUDIO_GUIDE.md (comprehensive guide)")
    print("\n🎭 Experience the power of emotional TTS!")
