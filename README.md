# Video Action Recognition (TimeSformer)

A small app that uses the pretrained TimeSformer (`facebook/timesformer-base-finetuned-k400`) to predict actions in your own short video clips (e.g., waving, playing guitar, basketball).

## Quickstart

### 1) Setup environment
```bash
# From the project directory
python3 -m venv .venv
source .venv/bin/activate  # on macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

If `decord` fails to install via wheels, install via Homebrew-provided ffmpeg and retry:
```bash
brew install ffmpeg
pip install decord --no-binary=:all:
```

### 2) Run CLI on a video
```bash
python predict.py /path/to/video.mp4 --top-k 5
```

### 3) Run Streamlit app
```bash
streamlit run app.py
```
Upload a short video and view top predictions.

## Notes
- Model: `facebook/timesformer-base-finetuned-k400` (Kinetics-400 labels)
- Inference uses uniformly sampled 32 frames via `decord`.
- Runs on GPU if available, otherwise CPU.
