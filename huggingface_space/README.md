---
title: Video Action Recognition
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 Video Action Recognition

An AI-powered video action recognition application using Facebook's TimeSformer model. Upload videos and get real-time predictions of human actions with confidence scores!

## 🚀 Features

- **🧠 AI-Powered**: Uses Facebook's TimeSformer model fine-tuned on Kinetics-400 dataset
- **⚡ GPU Accelerated**: Runs efficiently with GPU acceleration when available
- **📁 Easy Upload**: Simple drag-and-drop interface for video files
- **📊 Detailed Results**: Get top predictions with confidence scores and visual charts
- **🎯 400+ Actions**: Recognizes sports, daily activities, musical performances, and more

## 🎬 What Can It Recognize?

The model can identify hundreds of different actions including:

- **Sports**: Basketball, tennis, swimming, football, baseball, etc.
- **Daily Activities**: Cooking, cleaning, reading, writing, eating, etc.
- **Exercise & Fitness**: Yoga, running, weightlifting, cycling, dancing, etc.
- **Musical Performances**: Playing instruments, singing, conducting, etc.
- **Work Activities**: Typing, presenting, meeting, teaching, etc.
- **Social Interactions**: Waving, shaking hands, hugging, applauding, etc.

## 📖 How to Use

1. **Upload Video**: Click the upload area or drag and drop your video file
2. **Supported Formats**: MP4, MOV, AVI, MKV (MP4 recommended)
3. **File Size**: Keep videos under 100MB for optimal performance
4. **Duration**: 2-10 second clips work best
5. **Quality**: Use clear, well-lit videos with visible actions
6. **Get Results**: View predictions with confidence scores and visual chart

## 🛠️ Technical Details

- **Model**: `facebook/timesformer-base-finetuned-k400`
- **Dataset**: Kinetics-400 (400 action classes)
- **Input Processing**: 32 uniformly sampled frames per video
- **Architecture**: Vision Transformer for video understanding
- **Framework**: PyTorch + Transformers + Gradio

## 💡 Tips for Best Results

### Video Quality
- Use clear, well-lit videos
- Ensure actions are clearly visible
- Avoid overly shaky or blurry footage
- Keep subject in center of frame

### Video Length
- 2-10 seconds is optimal
- Longer videos are automatically sampled
- Focus on single, distinct actions

### Action Types
- Distinct, recognizable actions work best
- Sports activities tend to have high accuracy
- Daily activities are well-supported
- Subtle gestures may not be detected

## 📚 Model Information

**TimeSformer** is a state-of-the-art video understanding model that applies the Transformer architecture to video analysis. It processes videos by:

1. **Frame Extraction**: Samples 32 frames uniformly from input video
2. **Patch Embedding**: Converts frames into sequence of patches
3. **Temporal-Spatial Attention**: Applies attention across time and space
4. **Classification**: Predicts action from learned representations

### Performance
- **Top-1 Accuracy**: ~80% on Kinetics-400 test set
- **Parameters**: ~121M parameters
- **Input Resolution**: 224x224 pixels per frame
- **Inference Speed**: ~2-5 seconds per video (GPU)

## 🔗 Links & Resources

- **🏠 Project Homepage**: [Video Action Recognition](https://u-justine.github.io/VideoActionRecognition/)
- **💾 GitHub Repository**: [u-justine/VideoActionRecognition](https://github.com/u-justine/VideoActionRecognition)
- **🤗 Model Page**: [TimeSformer on Hugging Face](https://huggingface.co/facebook/timesformer-base-finetuned-k400)
- **📄 Research Paper**: [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- **📊 Dataset**: [Kinetics-400](https://deepmind.com/research/open-source/kinetics)

## 🚀 Run Locally

Want to run this on your own machine? Here's how:

### Quick Start
```bash
# Clone repository
git clone https://github.com/u-justine/VideoActionRecognition.git
cd VideoActionRecognition

# Setup environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Or run this Gradio version
cd huggingface_space
python app.py
```

### Google Colab
Try it in Google Colab with GPU acceleration:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/u-justine/VideoActionRecognition/blob/main/VideoActionRecognition_Colab.ipynb)

## 📝 Examples

### Supported Action Categories

| Sports | Daily Life | Exercise |
|--------|------------|----------|
| Playing basketball | Cooking | Doing yoga |
| Playing tennis | Reading | Running |
| Swimming | Writing | Weightlifting |
| Playing football | Eating | Cycling |
| Playing baseball | Cleaning | Dancing |

| Music & Arts | Work | Social |
|--------------|------|--------|
| Playing guitar | Typing | Waving |
| Playing piano | Presenting | Shaking hands |
| Dancing | Teaching | Applauding |
| Singing | Meeting | Hugging |
| Drawing | Working | Talking |

## 🐛 Troubleshooting

### Common Issues

**Video won't upload:**
- Check file format (MP4 recommended)
- Ensure file size < 100MB
- Try converting with online tools

**Poor predictions:**
- Use clearer, well-lit videos
- Ensure action is prominently visible
- Try shorter clips (2-10 seconds)
- Check if action is in supported categories

**Slow processing:**
- GPU acceleration improves speed significantly
- Smaller videos process faster
- Consider using local installation for frequent use

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/u-justine/VideoActionRecognition/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- **Facebook AI Research** for the TimeSformer model
- **Hugging Face** for the Transformers library and hosting
- **Google DeepMind** for the Kinetics dataset
- **Gradio** for the web interface framework

## 🌟 Star History

If you find this project useful, please consider giving it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=u-justine/VideoActionRecognition&type=Date)](https://star-history.com/#u-justine/VideoActionRecognition&Date)

---

**Built with ❤️ using TimeSformer, Transformers, and Gradio**

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/u-justine/VideoActionRecognition).