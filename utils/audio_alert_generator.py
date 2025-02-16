# audio_alert_generator.py
# Requires: `pip install -U simpleaudio numpy scipy`
# Usage:
#   - Play the sound: `python audio_alert_generator.py`
#   - Export to WAV:  `python audio_alert_generator.py --export pew-pew.wav`

import numpy as np
import simpleaudio as sa
import wave
import argparse

def generate_pew_pew(sample_rate=44100, duration=0.2, frequency=800, export_path=None):
    """Generates a 'pew-pew' sound and plays it. Optionally exports as a .wav file."""
    
    # Create a sine wave that "swoops" down in pitch (laser effect)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = np.sin(2 * np.pi * frequency * np.exp(-2 * t)) * 0.5  # Pitch drops over time

    # Convert to 16-bit PCM format
    audio = (wave_data * 32767).astype(np.int16)

    # Play the sound
    if export_path is None:
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
        play_obj.wait_done()
    else:
        save_as_wav(audio, sample_rate, export_path)

def save_as_wav(audio_data, sample_rate, filename):
    """Exports the generated sound as a .wav file."""
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"🔊 Exported: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 'pew-pew' sound. Optionally export as a .wav file.")
    parser.add_argument("--export", type=str, help="Path to save the .wav file instead of playing the sound.")
    args = parser.parse_args()

    generate_pew_pew(export_path=args.export)
