# AI Audio-to-Video Generator

A Python system that transforms audio tracks into cinematic videos. It uses audio analysis to detect emotion and tempo, and Stable Diffusion to generate matching imagery.

## Features

- **Audio Analysis**: Uses `librosa` to segment audio and detect beats, energy, and emotion (happy, sad, energetic, etc.).
- **Scene Planning**: Automatically maps audio segments to visual themes and prompts without an LLM.
- **Image Generation**: Uses Stable Diffusion (via `diffusers`) to create 8K UHD frames.
- **Video Rendering**: Compiles frames into a video synchronized with the audio using `moviepy`.

## Usage

```bash
python ai_video_generator.py --audio input.mp3 --style cinematic --theme nature
```

## Dependencies

- `librosa`
- `numpy`
- `moviepy`
- `diffusers`
- `torch`
- `opencv-python` (optional)
