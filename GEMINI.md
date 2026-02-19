# Voice Agent Project: GEMINI Context

This project is a multi-modal voice assistant composed of three primary AI components: Speech-to-Text (STT), a Large Language Model (LLM) for reasoning, and Text-to-Speech (TTS).

## Project Overview

The system is designed to facilitate a "Listen-Think-Speak" loop:
1.  **Whisper STT**: Converts user speech into text.
2.  **DeepSeek-R1**: Processes the text, performs reasoning, and generates a response.
3.  **KittenTTS**: Converts the generated text back into high-quality speech.

### Component Breakdown

*   **`whisper stt/`**: Contains the OpenAI Whisper codebase for robust multilingual speech recognition.
*   **`DeepSeek-R1/`**: Contains documentation and research papers for the DeepSeek-R1 reasoning model. Note: The model weights and inference engine are not included in this directory and must be sourced externally (e.g., via GGUF for local execution).
*   **`KittenTTS/`**: A lightweight, CPU-optimized ONNX-based TTS engine for real-time speech synthesis.

## Building and Running

### Prerequisites

*   **Python 3.8 - 3.11**: Recommended for compatibility across all models.
*   **FFmpeg**: **REQUIRED** for Whisper STT to process audio files. It must be installed and available in the system `PATH`.
*   **CUDA (Optional)**: While KittenTTS is CPU-optimized, Whisper and DeepSeek-R1 benefit significantly from GPU acceleration.

### Installation

1.  **Whisper STT**:
    ```bash
    cd "whisper stt"
    pip install -e .
    ```
2.  **KittenTTS**:
    ```bash
    cd KittenTTS
    pip install -e .
    ```
3.  **General Dependencies**:
    ```bash
    pip install sounddevice numpy
    ```

### Launching the Agent (TODO)

Currently, there is no top-level orchestrator script. To launch the integrated agent, a `main.py` needs to be developed to bridge these components.

## STT Component Status & Errors

During analysis of the `whisper stt` component, the following blockers were identified:

1.  **Missing FFmpeg**: The `ffmpeg` command-line tool is not found on the system. Whisper's `audio.py` relies on it for decoding and resampling.
2.  **Environment Setup**: The `whisper` module is not yet installed in the current Python environment.
3.  **Dependency Conflicts**: Ensure `triton` is only targeted for Linux environments as specified in `pyproject.toml`; Windows users should ignore triton errors during installation.

## Development Conventions

*   **Modular Architecture**: Each component is housed in its own directory with its own dependency management (`requirements.txt` or `pyproject.toml`).
*   **Local Inference**: The project prioritizes local execution over API-based services.
*   **Testing**: Test suites for Whisper are located in `whisper stt/tests/`. Run them using `pytest`.

## Usage

This directory serves as a workspace for integrating these three models. Developers should focus on creating a unified interface that captures microphone input, passes it through the Whisper-DeepSeek-Kitten pipeline, and outputs the result to speakers.
