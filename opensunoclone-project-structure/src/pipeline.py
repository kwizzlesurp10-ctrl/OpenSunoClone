
import torch
import torchaudio
import os
import time

# Import individual modules
from src.lyrics_gen import generate_lyrics
from src.music_gen import generate_instrumental
from src.vocals_gen import add_vocals
from src.stems import extract_stems

def create_song(prompt: str, duration: int = 60, weirdness: int = 0, output_format: str = 'wav') -> dict:
    """
    Orchestrates the end-to-end song generation process.
    
    This function sequences the lyrics, music, and vocal generation modules
    to produce a complete song from a single text prompt.
    
    Args:
        prompt: The creative prompt for the song (e.g., "upbeat pop about lost love").
        duration: The target duration of the song in seconds.
        weirdness: An integer (0-100) to vary the random seed for creative output.
        output_format: The desired output audio format (e.g., 'wav', 'mp3').
        
    Returns:
        A dictionary containing the path to the final song and its stems.
        
    Raises:
        RuntimeError: If any stage of the song generation process fails.
    """
    print("--- Starting Song Generation Pipeline ---")
    try:
        # --- 0. Set Random Seed for 'Weirdness' ---
        # Add a base seed for reproducibility and vary it with 'weirdness'
        seed = 42 + weirdness
        torch.manual_seed(seed)
        print(f"Using random seed: {seed}")

        # --- 1. Generate Lyrics ---
        print(f"[1/4] Generating lyrics for prompt: '{prompt}'...")
        lyrics = generate_lyrics(prompt, max_length=int(duration * 7))
        print("Lyrics generated successfully.")

        # --- 2. Generate Instrumental ---
        print(f"[2/4] Generating instrumental...")
        instrumental_path = generate_instrumental(prompt, duration=duration)
        print(f"Instrumental saved to: {instrumental_path}")

        # --- 3. Add Vocals and Mix ---
        print(f"[3/4] Synthesizing vocals and mixing track...")
        mixed_path = add_vocals(lyrics, instrumental_path)
        print(f"Vocals mixed successfully.")
        
        # --- 4. Final Polish, Save, and Extract Stems ---
        print("[4/4] Applying final touches and extracting stems...")
        waveform, sr = torchaudio.load(mixed_path)
        
        fade_out = torchaudio.functional.fade(waveform, fade_in_len=0, fade_out_len=sr * 2, fade_shape='linear')

        final_dir = "data/outputs"
        os.makedirs(final_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(final_dir, f"song_{timestamp}.{output_format}")
        
        torchaudio.save(final_path, fade_out, sr)
        print(f"Final song saved to: {final_path}")
        
        stems = extract_stems(final_path)
        print("Stems extracted successfully.")
        
        print(f"--- Pipeline Complete ---")
        
        return {
            'song_path': final_path,
            'stems': stems
        }

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"PIPELINE FAILED: {e}")
        raise RuntimeError(f"Create song error: {e}") from e

# Example for direct execution
if __name__ == "__main__":
    test_prompt = "upbeat pop about lost love"
    try:
        create_song(test_prompt, duration=15, weirdness=10)
    except RuntimeError as e:
        print(e)
