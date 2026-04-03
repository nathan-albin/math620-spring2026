import base64
import librosa
from IPython.display import HTML, display


def audio(y, sr, label="Play"):
    # Use librosa to write to a temporary buffer or a file
    import soundfile as sf
    import io

    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    # Create a clean HTML button with an invisible audio element
    audio_id = f"audio_{hash(label)}"
    html = f"""
    <div style="margin: 10px 0;">
        <audio id="{audio_id}" src="data:audio/wav;base64,{b64}"></audio>
        <button onclick="document.getElementById('{audio_id}').play()" 
                aria-label="Play {label} audio"
                style="padding: 10px 20px; cursor: pointer;">
            Play {label}
        </button>
    </div>
    """
    display(HTML(html))
