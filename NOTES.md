# Run Skill Development Notes

## Project: IndexTTS2
A text-to-speech system with emotional control and voice cloning capabilities.

## Discovery Phase

### What is this?
- **Type**: Web-based TTS application (Gradio WebUI) + API server (FastAPI)
- **Main entry points**:
  - `webui.py` - Gradio web interface on port 7860
  - `api_server.py` - FastAPI REST API
  - CLI tools: `indextts` and `indextts2` commands

### Project Structure
- Uses `uv` for dependency management
- Python 3.10+ required
- PyTorch 2.8 with CUDA 12.8
- Models downloaded from HuggingFace/ModelScope
- Checkpoints required: `gpt.pth`, `s2mel.pth`, `bpe.model`, etc.

### Dependencies Status
- No package.json (not a Node.js project)
- Uses pyproject.toml with uv
- Models appear to be in checkpoints/ but incomplete (only config files present)

## Execution Phase

### Next Steps
1. Check if virtual environment exists
2. Install dependencies with `uv sync --all-extras`
3. Download models if missing
4. Launch webui and test with chromium-cli
5. Take screenshots and build interaction harness
