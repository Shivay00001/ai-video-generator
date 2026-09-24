#!/usr/bin/env python3
"""Entry point for the AI Audio-to-Video Generator.

Wires the real assembler (ai_video_generator.AudioToVideoGenerator) to the
real env-key LLM client (llm_client.LLMClient) for AI scene-script generation.
Set OPENAI_API_KEY (optionally OPENAI_BASE_URL / OPENAI_MODEL) to enable it;
without a key the generator falls back to static style mappings.

Usage:
    python main.py --audio input.wav --output out.mp4 --mock
"""
import argparse

from ai_video_generator import AudioToVideoGenerator, SCENE_THEMES
from llm_client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="AI Audio-to-Video Generator")
    parser.add_argument("--audio", required=True, help="Input audio file (WAV recommended)")
    parser.add_argument("--output", default="output.mp4", help="Output video file")
    parser.add_argument("--style", default="cinematic", help="Video style")
    parser.add_argument("--theme", default="nature", choices=list(SCENE_THEMES.keys()),
                        help="Scene theme")
    parser.add_argument("--segment-duration", type=float, default=3.0,
                        help="Duration of each scene in seconds")
    parser.add_argument("--subtitles", help="Optional SRT subtitle file")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock image generation (for testing, no Stable Diffusion)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Force static style mappings even if an API key is set")
    args = parser.parse_args()

    llm_client = None if args.no_llm else LLMClient()
    if llm_client is not None and not llm_client.configured:
        print("Note: OPENAI_API_KEY not set - using static style mappings for scene scripts.")

    generator = AudioToVideoGenerator(
        style=args.style, theme=args.theme, use_mock=args.mock, llm_client=llm_client)
    generator.generate(
        audio_path=args.audio,
        output_path=args.output,
        segment_duration=args.segment_duration,
        subtitles_path=args.subtitles,
    )


if __name__ == "__main__":
    main()
