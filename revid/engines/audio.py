# -*- coding: utf-8 -*-

"""Audio AI engines: Demucs, Silero, RNNoise, Open-Unmix, AudioSR, AERO.

Audio engines operate on audio files, not frame directories.
The render pipeline extracts audio → processes → remuxes.
"""

import glob
import os
import subprocess

from revid.engines.registry import register


def _get_audio_path(input_dir: str) -> str:
    """Find the audio file in the input directory.

    Audio engines receive the same input_dir as video engines,
    but the render pipeline places an audio.wav file there for audio steps.
    """
    for ext in ["*.wav", "*.mp3", "*.flac", "*.aac"]:
        files = glob.glob(os.path.join(input_dir, ext))
        if files:
            return files[0]
    # If no audio file, check if frames exist (wrong step type)
    raise FileNotFoundError(f"No audio file found in {input_dir}")


# =============================================================================
# Audio denoise
# =============================================================================


@register("audio_denoise", "demucs")
def audio_denoise_demucs(step: dict, input_dir: str, output_dir: str) -> None:
    """Denoise audio using Demucs (Meta) — separate sources, keep speech."""
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError("Demucs not found. Install with:\n  pip install demucs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = step.get("model", "htdemucs")

    model = get_model(model_name).to(device)
    model.eval()

    audio_path = _get_audio_path(input_dir)
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]

    # sources: [drums, bass, other, vocals] — keep vocals + other (remove noise)
    # For denoise: mix vocals + other, discard drums/bass if they're noise
    vocals_idx = model.sources.index("vocals")
    other_idx = model.sources.index("other")
    cleaned = sources[vocals_idx] + sources[other_idx]

    out_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(out_path, cleaned.cpu(), sample_rate)


@register("audio_denoise", "silero")
def audio_denoise_silero(step: dict, input_dir: str, output_dir: str) -> None:
    """Denoise audio using Silero VAD + noise gating."""
    try:
        import torch
        import torchaudio
    except ImportError:
        raise ImportError("Silero requires PyTorch. Install with:\n  pip install torch torchaudio")

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )

    audio_path = _get_audio_path(input_dir)
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz for VAD
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        wav_16k = resampler(waveform)
    else:
        wav_16k = waveform

    # Get speech timestamps
    get_speech_timestamps = utils[0]
    speech_timestamps = get_speech_timestamps(wav_16k[0], model, sampling_rate=16000)

    # Create mask and apply noise gate
    ratio = sample_rate / 16000
    mask = torch.zeros_like(waveform[0])
    for ts in speech_timestamps:
        start = int(ts["start"] * ratio)
        end = int(ts["end"] * ratio)
        mask[start:end] = 1.0

    # Smooth mask edges to avoid clicks
    kernel_size = int(sample_rate * 0.01)
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = torch.nn.functional.avg_pool1d(
        mask.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2
    ).squeeze()

    cleaned = waveform * mask.unsqueeze(0)

    out_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(out_path, cleaned, sample_rate)


@register("audio_denoise", "rnnoise")
def audio_denoise_rnnoise(step: dict, input_dir: str, output_dir: str) -> None:
    """Denoise audio using RNNoise (lightweight, CPU-based)."""
    try:
        import rnnoise
    except ImportError:
        raise ImportError("RNNoise not found. Install with:\n  pip install rnnoise-python")

    import wave

    import numpy as np

    audio_path = _get_audio_path(input_dir)

    # RNNoise works on 48kHz mono 16-bit PCM
    # Convert input to WAV 48kHz mono first
    temp_wav = os.path.join(output_dir, "temp_48k.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            temp_wav,
        ],
        check=True,
    )

    # Process with RNNoise
    denoiser = rnnoise.RNNoise()

    with wave.open(temp_wav, "rb") as wf:
        wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

    # Process in 10ms frames (480 samples at 48kHz)
    frame_size = 480
    output_data = np.zeros_like(audio_data)

    for i in range(0, len(audio_data) - frame_size, frame_size):
        frame = audio_data[i : i + frame_size].astype(np.float32)
        denoised = denoiser.process_frame(frame)
        output_data[i : i + frame_size] = np.array(denoised, dtype=np.int16)

    out_path = os.path.join(output_dir, "audio.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(output_data.tobytes())

    os.remove(temp_wav)


# =============================================================================
# Audio source separation
# =============================================================================


@register("audio_separate", "demucs")
def audio_separate_demucs(step: dict, input_dir: str, output_dir: str) -> None:
    """Separate audio sources using Demucs (vocals, drums, bass, other)."""
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError("Demucs not found. Install with:\n  pip install demucs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = step.get("model", "htdemucs")
    stem = step.get("stem", "vocals")  # which stem to keep

    model = get_model(model_name).to(device)
    model.eval()

    audio_path = _get_audio_path(input_dir)
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]

    stem_idx = model.sources.index(stem)
    out_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(out_path, sources[stem_idx].cpu(), sample_rate)


@register("audio_separate", "open_unmix")
def audio_separate_open_unmix(step: dict, input_dir: str, output_dir: str) -> None:
    """Separate audio sources using Open-Unmix."""
    try:
        import torch
        import torchaudio
        from openunmix import predict
    except ImportError:
        raise ImportError("Open-Unmix not found. Install with:\n  pip install openunmix")

    stem = step.get("stem", "vocals")
    model_name = step.get("model", "umxhq")

    audio_path = _get_audio_path(input_dir)
    waveform, sample_rate = torchaudio.load(audio_path)

    estimates = predict.separate(
        audio=waveform.unsqueeze(0),
        rate=sample_rate,
        model_str_or_path=model_name,
    )

    out_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(out_path, estimates[stem][0], sample_rate)


# =============================================================================
# Audio super resolution
# =============================================================================


@register("audio_upscale", "audiosr")
def audio_upscale_audiosr(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale audio bandwidth using AudioSR."""
    try:
        import torch
        from audiosr import build_model, super_resolution
    except ImportError:
        raise ImportError(
            "AudioSR not found. Install with:\n"
            "  pip install audiosr\n"
            "  Or clone https://github.com/haoheliu/versatile_audio_super_resolution"
        )

    model_name = step.get("model", "basic")
    audio_path = _get_audio_path(input_dir)

    audiosr_model = build_model(model_name=model_name)

    waveform = super_resolution(
        audiosr_model,
        audio_path,
        seed=42,
        guidance_scale=3.5,
        ddim_steps=50,
    )

    import torchaudio

    out_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(out_path, torch.tensor(waveform).float(), 48000)


@register("audio_upscale", "aero")
def audio_upscale_aero(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale audio bandwidth using AERO (Facebook Research)."""
    try:
        import torch
        import torchaudio
    except ImportError:
        raise ImportError("AERO requires PyTorch. Install with: pip install torch torchaudio")

    try:
        from aero.model import AERO
        from basicsr.utils.download_util import load_file_from_url

        model_path = load_file_from_url(
            url="https://github.com/facebookresearch/aero/releases/download/v1.0/aero.pth",
            model_dir="weights/AERO",
            progress=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AERO().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except ImportError:
        raise ImportError("AERO not found. Clone from:\n  https://github.com/facebookresearch/aero")

    audio_path = _get_audio_path(input_dir)
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        output = model(waveform.unsqueeze(0))

    out_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(out_path, output[0].cpu(), 48000)
