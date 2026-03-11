
import pytest
import os
import shutil
from unittest.mock import patch, MagicMock
from src.pipeline import create_song

# This is a unit test that mocks the expensive AI calls.
# It verifies that the pipeline correctly calls each module in sequence
# and handles the file paths correctly.

@patch('torch.manual_seed')
@patch('src.pipeline.generate_lyrics')
@patch('src.pipeline.generate_instrumental')
@patch('src.pipeline.add_vocals')
@patch('src.pipeline.extract_stems')
@patch('torchaudio.load')
@patch('torchaudio.save')
def test_pipeline_orchestration(
    mock_save, mock_load, mock_extract_stems, mock_add_vocals, mock_generate_instrumental, mock_generate_lyrics, mock_manual_seed
):
    """
    Tests the pipeline's orchestration logic without running the actual models.
    """
    # --- Setup Mocks ---
    mock_generate_lyrics.return_value = "Mocked lyrics."
    mock_generate_instrumental.return_value = "/mock/instrumental.wav"
    mock_add_vocals.return_value = "/mock/mixed.wav"
    mock_extract_stems.return_value = {
        'lows': '/mock/stems/lows.wav',
        'highs': '/mock/stems/highs.wav'
    }
    
    # Mock the waveform and its fade method
    mock_waveform = MagicMock()
    mock_fade_waveform = MagicMock()
    mock_waveform.fade.return_value = mock_fade_waveform
    mock_load.return_value = (mock_waveform, 44100)
    
    # --- Execute ---
    prompt = "upbeat pop song about testing"
    result = create_song(prompt, duration=10, weirdness=5, output_format='wav')
    
    # --- Assertions ---
    mock_manual_seed.assert_called_once_with(42 + 5)
    mock_generate_lyrics.assert_called_once_with(prompt, max_length=70)
    mock_generate_instrumental.assert_called_once_with(prompt, duration=10)
    mock_add_vocals.assert_called_once_with("Mocked lyrics.", "/mock/instrumental.wav")
    mock_load.assert_called_once_with("/mock/mixed.wav")
    mock_save.assert_called_once()
    mock_extract_stems.assert_called_once_with(mock_save.call_args[0][0]) # Check it's called with the final path
    
    assert 'song_path' in result
    assert result['song_path'].startswith("data/outputs/song_")
    assert result['stems']['lows'] == '/mock/stems/lows.wav'

@patch('src.pipeline.generate_lyrics', side_effect=RuntimeError("Model download failed"))
def test_pipeline_handles_lyrics_failure(mock_generate_lyrics):
    """Tests that the pipeline gracefully fails if a sub-module raises an error."""
    with pytest.raises(RuntimeError, match="Create song error: Model download failed"):
        create_song("any prompt")


# To run full integration test: pytest -m full_integration
@pytest.mark.full_integration
def test_full_pipeline_execution():
    """
    A full end-to-end integration test.
    WARNING: This test is extremely slow, downloads models, and uses significant resources.
    It should be run manually or on a dedicated CI runner.
    """
    prompt = "sad rock song about a broken computer"
    result = None
    created_files = []
    try:
        result = create_song(prompt, duration=5, weirdness=1)
        
        # Verify song path
        song_path = result['song_path']
        created_files.append(song_path)
        assert os.path.exists(song_path)
        
        # Verify stems
        assert 'lows' in result['stems']
        lows_path = result['stems']['lows']
        created_files.append(lows_path)
        assert os.path.exists(lows_path)
        
    finally:
        # Comprehensive cleanup
        for path in created_files:
            if os.path.exists(path):
                os.remove(path)
        # Clean up directories that might have been created by other modules
        if os.path.exists('data/outputs/instrumentals'):
             shutil.rmtree('data/outputs/instrumentals')
        if os.path.exists('data/outputs/lyrics'):
             shutil.rmtree('data/outputs/lyrics')
        if os.path.exists('data/outputs/mixed'):
             shutil.rmtree('data/outputs/mixed')
