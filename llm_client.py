"""Real LLM client for AI script/prompt generation.

Reads the API key from the environment (OPENAI_API_KEY). The endpoint is
OpenAI-compatible and overridable via OPENAI_BASE_URL (works with OpenRouter,
Groq, etc.), model via OPENAI_MODEL.

There is no mock fallback: without a key, generate_scene_prompts() raises a
clear RuntimeError. With a key it makes a real chat-completions HTTP call.
"""
import json
import os
import requests

REQUEST_TIMEOUT = 90


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL",
                         "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def configured(self):
        return bool(self.api_key) and self.api_key != "sk-placeholder"

    def generate_scene_prompts(self, segments_summary, style, theme, count):
        """Generate one image prompt (+ negative) per scene via a real LLM call.

        Returns a list of {"prompt": ..., "negative_prompt": ...} dicts.
        """
        if not self.configured:
            raise RuntimeError(
                "No LLM API key configured. Set OPENAI_API_KEY in the environment "
                "(OPENAI_BASE_URL / OPENAI_MODEL optional) to enable AI script generation."
            )
        system = (
            "You are a cinematic director. For each audio segment described, write one "
            "detailed Stable Diffusion image prompt and one negative prompt. "
            "Reply with a JSON array of objects: "
            '[{"prompt": "...", "negative_prompt": "..."}]. JSON only, no markdown.'
        )
        user = (
            f"Style: {style}. Theme: {theme}. Generate exactly {count} scene prompts.\n"
            f"Audio segments (start-end seconds, emotion, energy, tempo):\n{segments_summary}"
        )
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.8},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        # Tolerate ```json fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
        scenes = json.loads(text)
        if not isinstance(scenes, list) or len(scenes) != count:
            raise RuntimeError(
                f"LLM returned {len(scenes) if isinstance(scenes, list) else 'non-list'} "
                f"scene prompts, expected {count}")
        return [{"prompt": s["prompt"], "negative_prompt": s.get("negative_prompt", "")}
                for s in scenes]
