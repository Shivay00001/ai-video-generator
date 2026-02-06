"""
AI Audio-to-Video Generator
Converts audio (songs/narration) into cinematic videos with Stable Diffusion
NO LLM REQUIRED - Uses static style mappings and audio analysis
"""

import os
import librosa
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
from dataclasses import dataclass
import subprocess

# Video/Image processing
try:
    from moviepy.editor import (VideoFileClip, AudioFileClip, ImageClip, 
                                concatenate_videoclips, CompositeVideoClip, TextClip)
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("Warning: moviepy not available")
    MOVIEPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: opencv-python not available")
    CV2_AVAILABLE = False

# Stable Diffusion
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available - image generation disabled")
    DIFFUSERS_AVAILABLE = False


# ============================================================================
# STYLE AND EMOTION MAPPINGS (No LLM Required)
# ============================================================================

STYLE_MAP = {
    "happy": {
        "prompt": "bright cinematic scene, sunshine, open fields, vibrant colors, joyful atmosphere, golden hour lighting, professional photography, 8k uhd",
        "negative": "dark, sad, gloomy, low quality, blurry",
        "color_grade": "warm"
    },
    "sad": {
        "prompt": "melancholic cinematic scene, rainy streets, muted tones, emotional lighting, moody atmosphere, professional cinematography, 8k uhd",
        "negative": "bright, cheerful, colorful, low quality, blurry",
        "color_grade": "cool"
    },
    "energetic": {
        "prompt": "dynamic cinematic scene, urban nightlife, vivid neon colors, fast motion blur, exciting atmosphere, professional photography, 8k uhd",
        "negative": "static, boring, dull, low quality, blurry",
        "color_grade": "vibrant"
    },
    "calm": {
        "prompt": "peaceful cinematic scene, serene landscape, soft lighting, tranquil atmosphere, pastel colors, professional photography, 8k uhd",
        "negative": "chaotic, busy, intense, low quality, blurry",
        "color_grade": "soft"
    },
    "dramatic": {
        "prompt": "intense cinematic scene, dramatic lighting, high contrast, epic atmosphere, theatrical composition, professional cinematography, 8k uhd",
        "negative": "flat, boring, simple, low quality, blurry",
        "color_grade": "high_contrast"
    },
    "romantic": {
        "prompt": "romantic cinematic scene, soft focus, warm lighting, intimate atmosphere, beautiful bokeh, professional photography, 8k uhd",
        "negative": "harsh, cold, distant, low quality, blurry",
        "color_grade": "warm_soft"
    }
}

SCENE_THEMES = {
    "nature": ["mountain landscape", "ocean waves", "forest path", "sunset sky", "flower field"],
    "urban": ["city skyline", "street scene", "modern architecture", "busy intersection", "neon lights"],
    "abstract": ["flowing colors", "geometric patterns", "light particles", "cosmic nebula", "digital art"],
    "people": ["silhouette figure", "dancing crowd", "portrait close-up", "group celebration", "solo artist"],
    "cinematic": ["wide angle shot", "aerial view", "tracking shot", "close-up detail", "establishing shot"]
}


@dataclass
class AudioSegment:
    """Represents a segment of audio with emotional characteristics"""
    start_time: float
    end_time: float
    duration: float
    energy: float
    tempo: float
    emotion: str
    beat_strength: float
    spectral_centroid: float


@dataclass
class VideoScene:
    """Represents a video scene to be generated"""
    segment: AudioSegment
    prompt: str
    negative_prompt: str
    style: str
    theme: str
    transition: str


# ============================================================================
# AUDIO ANALYZER
# ============================================================================

class AudioAnalyzer:
    """Analyzes audio to detect beats, tempo, emotion, and energy"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
    
    def analyze(self, audio_path: str, segment_duration: float = 3.0) -> List[AudioSegment]:
        """
        Analyze audio and segment into scenes
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of AudioSegment objects
        """
        print(f"Analyzing audio: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (excitement)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Create segments
        segments = []
        num_segments = int(np.ceil(duration / segment_duration))
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            # Get feature values for this segment
            start_frame = int(start_time * sr / 512)
            end_frame = int(end_time * sr / 512)
            
            segment_energy = np.mean(rms[start_frame:end_frame])
            segment_centroid = np.mean(spectral_centroid[start_frame:end_frame])
            segment_zcr = np.mean(zcr[start_frame:end_frame])
            
            # Detect emotion based on audio features
            emotion = self._detect_emotion(
                segment_energy, segment_centroid, segment_zcr, tempo
            )
            
            # Beat strength in this segment
            beats_in_segment = beats[(beats >= start_frame) & (beats < end_frame)]
            beat_strength = len(beats_in_segment) / (end_time - start_time)
            
            segment = AudioSegment(
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                energy=float(segment_energy),
                tempo=float(tempo),
                emotion=emotion,
                beat_strength=float(beat_strength),
                spectral_centroid=float(segment_centroid)
            )
            
            segments.append(segment)
        
        print(f"Created {len(segments)} segments")
        return segments
    
    def _detect_emotion(self, energy: float, centroid: float, zcr: float, tempo: float) -> str:
        """
        Detect emotion from audio features
        Simple heuristic-based classification
        """
        # Normalize features (rough thresholds)
        energy_norm = min(energy / 0.1, 1.0)
        centroid_norm = min(centroid / 3000, 1.0)
        zcr_norm = min(zcr / 0.15, 1.0)
        tempo_norm = min(tempo / 180, 1.0)
        
        # Decision tree for emotion
        if energy_norm > 0.7 and tempo_norm > 0.7:
            return "energetic"
        elif energy_norm > 0.6 and centroid_norm > 0.6:
            return "happy"
        elif energy_norm < 0.3 and centroid_norm < 0.4:
            return "sad"
        elif energy_norm < 0.4 and tempo_norm < 0.5:
            return "calm"
        elif energy_norm > 0.5 and centroid_norm > 0.5:
            return "dramatic"
        else:
            return "calm"


# ============================================================================
# SCENE PLANNER
# ============================================================================

class ScenePlanner:
    """Plans video scenes based on audio segments"""
    
    def __init__(self, style: str = "cinematic", theme: str = "nature"):
        self.style = style
        self.theme = theme
        self.scene_index = 0
    
    def plan_scenes(self, segments: List[AudioSegment]) -> List[VideoScene]:
        """
        Plan video scenes for each audio segment
        
        Args:
            segments: List of audio segments
            
        Returns:
            List of VideoScene objects
        """
        print(f"Planning {len(segments)} scenes...")
        
        scenes = []
        theme_variations = SCENE_THEMES.get(self.theme, SCENE_THEMES["cinematic"])
        
        for i, segment in enumerate(segments):
            # Get emotion-based style
            emotion = segment.emotion
            style_config = STYLE_MAP.get(emotion, STYLE_MAP["calm"])
            
            # Cycle through theme variations
            theme_element = theme_variations[i % len(theme_variations)]
            
            # Build prompt
            prompt = f"{theme_element}, {style_config['prompt']}"
            negative_prompt = style_config['negative']
            
            # Determine transition
            if i == 0:
                transition = "fade_in"
            elif i == len(segments) - 1:
                transition = "fade_out"
            elif segment.energy > segments[i-1].energy * 1.3:
                transition = "cut"  # Energetic transition
            else:
                transition = "dissolve"  # Smooth transition
            
            scene = VideoScene(
                segment=segment,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style=emotion,
                theme=theme_element,
                transition=transition
            )
            
            scenes.append(scene)
        
        print(f"Scene planning complete")
        return scenes


# ============================================================================
# IMAGE GENERATOR (Stable Diffusion)
# ============================================================================

class ImageGenerator:
    """Generates images using Stable Diffusion"""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", 
                 device: str = "cuda", use_mock: bool = False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_mock = use_mock or not DIFFUSERS_AVAILABLE
        
        if not self.use_mock:
            print(f"Loading Stable Diffusion on {self.device}...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            print("Stable Diffusion loaded")
        else:
            print("Using mock image generation (colored gradients)")
            self.pipe = None
    
    def generate(self, prompt: str, negative_prompt: str, 
                 width: int = 1024, height: int = 576, 
                 num_inference_steps: int = 20, seed: int = None) -> np.ndarray:
        """
        Generate image from prompt
        
        Returns:
            numpy array (H, W, 3) in RGB format
        """
        if self.use_mock:
            return self._generate_mock(prompt, width, height)
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
        
        return np.array(image)
    
    def _generate_mock(self, prompt: str, width: int, height: int) -> np.ndarray:
        """Generate mock gradient image (for testing without GPU)"""
        # Create gradient based on prompt keywords
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color based on emotion keywords
        if "bright" in prompt or "happy" in prompt:
            color = [255, 200, 100]  # Warm yellow
        elif "sad" in prompt or "rain" in prompt:
            color = [100, 120, 150]  # Cool blue-gray
        elif "energetic" in prompt or "neon" in prompt:
            color = [255, 50, 150]  # Vibrant magenta
        elif "calm" in prompt or "peaceful" in prompt:
            color = [120, 180, 140]  # Soft green
        else:
            color = [150, 150, 150]  # Neutral gray
        
        # Create gradient
        for i in range(height):
            factor = i / height
            img[i, :] = [int(c * (0.5 + factor * 0.5)) for c in color]
        
        # Add text
        if CV2_AVAILABLE:
            cv2.putText(img, "MOCK IMAGE", (width//4, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(img, prompt[:50], (50, height-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img


# ============================================================================
# VIDEO RENDERER
# ============================================================================

class VideoRenderer:
    """Renders final video with audio sync"""
    
    def __init__(self, fps: int = 30, resolution: Tuple[int, int] = (1920, 1080)):
        self.fps = fps
        self.resolution = resolution
    
    def render(self, scenes: List[VideoScene], image_generator: ImageGenerator,
               audio_path: str, output_path: str, 
               subtitles: Optional[List[Dict]] = None) -> str:
        """
        Render complete video
        
        Args:
            scenes: List of video scenes
            image_generator: Image generator instance
            audio_path: Path to audio file
            output_path: Output video path
            subtitles: Optional list of subtitle dicts with 'start', 'end', 'text'
            
        Returns:
            Path to rendered video
        """
        print(f"Rendering {len(scenes)} scenes to video...")
        
        if not MOVIEPY_AVAILABLE:
            print("Error: moviepy not available")
            return None
        
        # Generate images for each scene
        temp_dir = tempfile.mkdtemp()
        video_clips = []
        
        for i, scene in enumerate(scenes):
            print(f"Generating scene {i+1}/{len(scenes)}: {scene.theme} ({scene.style})")
            
            # Generate image
            img = image_generator.generate(
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                width=self.resolution[0],
                height=self.resolution[1],
                seed=i * 1000
            )
            
            # Save image
            img_path = os.path.join(temp_dir, f"scene_{i:04d}.png")
            if CV2_AVAILABLE:
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                from PIL import Image
                Image.fromarray(img).save(img_path)
            
            # Create video clip
            clip = ImageClip(img_path).set_duration(scene.segment.duration)
            
            # Apply transition
            if scene.transition == "fade_in":
                clip = clip.fadein(0.5)
            elif scene.transition == "fade_out":
                clip = clip.fadeout(0.5)
            
            # Apply zoom effect based on energy
            if scene.segment.energy > 0.08:
                clip = clip.resize(lambda t: 1 + 0.1 * t / scene.segment.duration)
            
            video_clips.append(clip)
        
        # Concatenate all clips
        print("Concatenating clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        # Add audio
        print("Adding audio...")
        audio = AudioFileClip(audio_path)
        final_video = final_video.set_audio(audio)
        
        # Add subtitles if provided
        if subtitles:
            print("Adding subtitles...")
            final_video = self._add_subtitles(final_video, subtitles)
        
        # Write final video
        print(f"Writing video to {output_path}...")
        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(temp_dir, 'temp_audio.m4a'),
            remove_temp=True,
            threads=4
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"Video rendered successfully: {output_path}")
        return output_path
    
    def _add_subtitles(self, video: VideoFileClip, subtitles: List[Dict]) -> CompositeVideoClip:
        """Add subtitles to video"""
        subtitle_clips = []
        
        for sub in subtitles:
            txt_clip = TextClip(
                sub['text'],
                fontsize=40,
                color='white',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(video.w * 0.8, None)
            )
            txt_clip = txt_clip.set_position(('center', 0.85), relative=True)
            txt_clip = txt_clip.set_start(sub['start'])
            txt_clip = txt_clip.set_duration(sub['end'] - sub['start'])
            
            subtitle_clips.append(txt_clip)
        
        return CompositeVideoClip([video] + subtitle_clips)


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class AudioToVideoGenerator:
    """Complete audio-to-video generation system"""
    
    def __init__(self, style: str = "cinematic", theme: str = "nature",
                 use_mock: bool = False):
        """
        Initialize generator
        
        Args:
            style: Video style (cinematic, artistic, etc)
            theme: Scene theme (nature, urban, abstract, etc)
            use_mock: Use mock image generation (for testing)
        """
        self.analyzer = AudioAnalyzer()
        self.planner = ScenePlanner(style=style, theme=theme)
        self.image_generator = ImageGenerator(use_mock=use_mock)
        self.renderer = VideoRenderer()
    
    def generate(self, audio_path: str, output_path: str = "output.mp4",
                 segment_duration: float = 3.0,
                 subtitles_path: Optional[str] = None) -> str:
        """
        Generate video from audio
        
        Args:
            audio_path: Path to audio file
            output_path: Output video path
            segment_duration: Duration of each scene in seconds
            subtitles_path: Optional path to SRT subtitle file
            
        Returns:
            Path to generated video
        """
        print("=" * 70)
        print("AI AUDIO-TO-VIDEO GENERATOR")
        print("=" * 70)
        
        # Step 1: Analyze audio
        print("\n[1/4] Analyzing audio...")
        segments = self.analyzer.analyze(audio_path, segment_duration)
        
        # Step 2: Plan scenes
        print("\n[2/4] Planning scenes...")
        scenes = self.planner.plan_scenes(segments)
        
        # Step 3: Load subtitles if provided
        subtitles = None
        if subtitles_path and os.path.exists(subtitles_path):
            print("\n[3/4] Loading subtitles...")
            subtitles = self._load_subtitles(subtitles_path)
        else:
            print("\n[3/4] No subtitles provided")
        
        # Step 4: Render video
        print("\n[4/4] Rendering video...")
        result = self.renderer.render(
            scenes=scenes,
            image_generator=self.image_generator,
            audio_path=audio_path,
            output_path=output_path,
            subtitles=subtitles
        )
        
        # Save metadata
        metadata = self._generate_metadata(audio_path, scenes, output_path)
        metadata_path = output_path.replace('.mp4', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE!")
        print(f"Video: {output_path}")
        print(f"Metadata: {metadata_path}")
        print("=" * 70)
        
        return result
    
    def _load_subtitles(self, srt_path: str) -> List[Dict]:
        """Load subtitles from SRT file"""
        subtitles = []
        # Simple SRT parser
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic parsing (simplified)
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                # Parse timing
                time_line = lines[1]
                if '-->' in time_line:
                    start, end = time_line.split('-->')
                    start_sec = self._srt_time_to_seconds(start.strip())
                    end_sec = self._srt_time_to_seconds(end.strip())
                    text = ' '.join(lines[2:])
                    
                    subtitles.append({
                        'start': start_sec,
                        'end': end_sec,
                        'text': text
                    })
        
        return subtitles
    
    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format to seconds"""
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def _generate_metadata(self, audio_path: str, scenes: List[VideoScene], 
                          output_path: str) -> Dict:
        """Generate metadata for the video"""
        return {
            'audio_source': audio_path,
            'output_video': output_path,
            'num_scenes': len(scenes),
            'total_duration': sum(s.segment.duration for s in scenes),
            'scenes': [
                {
                    'index': i,
                    'start': s.segment.start_time,
                    'end': s.segment.end_time,
                    'emotion': s.segment.emotion,
                    'energy': s.segment.energy,
                    'theme': s.theme,
                    'prompt': s.prompt
                }
                for i, s in enumerate(scenes)
            ],
            'copyright': 'AI-generated video - copyright-free',
            'generated_with': 'Audio-to-Video AI Generator (No LLM)'
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Audio-to-Video Generator")
    parser.add_argument('--audio', required=True, help='Input audio file (MP3/WAV)')
    parser.add_argument('--output', default='output.mp4', help='Output video file')
    parser.add_argument('--style', default='cinematic', help='Video style')
    parser.add_argument('--theme', default='nature', 
                       choices=list(SCENE_THEMES.keys()),
                       help='Scene theme')
    parser.add_argument('--segment-duration', type=float, default=3.0,
                       help='Duration of each scene in seconds')
    parser.add_argument('--subtitles', help='Optional SRT subtitle file')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock image generation (for testing)')
    
    args = parser.parse_args()
    
    # Generate video
    generator = AudioToVideoGenerator(
        style=args.style,
        theme=args.theme,
        use_mock=args.mock
    )
    
    generator.generate(
        audio_path=args.audio,
        output_path=args.output,
        segment_duration=args.segment_duration,
        subtitles_path=args.subtitles
    )
