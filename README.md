# Ai Video Generator

AI-powered video generator with web UI, Docker & CI

![Language](https://img.shields.io/badge/Language-Python-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Overview

Welcome to the **Ai Video Generator** repository. This project is built to deliver a robust and scalable solution tailored to modern development standards.

## ✨ Features

- **High Performance:** Optimized for speed and efficiency.
- **Scalable Architecture:** Designed to grow with your needs.
- **Clean Codebase:** Follows best practices and industry standards.
- **Secure by Default:** Engineered with security in mind.

## 🛠️ Prerequisites

Ensure you have the following installed in your environment before proceeding:
- Appropriate runtime/compiler for `Python`
- Standard development tools

## 📦 Installation

Follow standard installation steps for `Python` to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivay00001/ai-video-generator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-video-generator
   ```
3. Install dependencies according to the standard `Python` ecosystem.

## 💻 Usage

Run the project using standard execution commands for `Python`. Ensure all environment variables and configurations are set prior to execution.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is licensed under standard terms.

---

## 🔧 Wave-1 fix notes (2026-09-24)

- **Real LLM client:** new `llm_client.py` — reads `OPENAI_API_KEY` from the environment
  (`OPENAI_BASE_URL`/`OPENAI_MODEL` optional, so it also works with OpenRouter/Groq),
  and makes a **real** chat-completions HTTP call to generate one image prompt +
  negative prompt per scene. No mock fallback: without a key it raises a clear
  `RuntimeError`. `ScenePlanner.plan_scenes()` / `AudioToVideoGenerator` accept an
  `llm_client`; when it is configured, prompts come from the LLM, otherwise the
  original static style mappings are used (stated honestly at runtime).
- **Real audio analysis without librosa:** `librosa` is now optional. For WAV files a
  built-in fallback analyzer (stdlib `wave` + numpy) computes real RMS energy,
  spectral centroid (FFT) and zero-crossing rate per segment.
- **Bug fixes:** module crashed at import when `moviepy`/`diffusers` were missing
  (annotations and `torch.cuda` referenced unconditionally); now guarded.
- **Entry point:** new `main.py` CLI wiring the assembler to the LLM client
  (`Dockerfile CMD ["python", "main.py"]` now points at a file that exists).
- Verified 2026-09-24 on a real synthesized 6 s WAV: 2 segments analyzed with real
  features, 2 scenes planned (static path — no key in sandbox), LLM client raises
  the documented no-key error and returns HTTP 401 with a dummy key (proving a real
  API call, not a stub), mock image gen produces a valid array.
- **What still needs a key/install:** `OPENAI_API_KEY` for AI scripts; `librosa`
  for MP3/richer features; `moviepy` + ffmpeg for actual video rendering;
  `diffusers` + torch + GPU for real Stable Diffusion images (`--mock` otherwise).
