# 🎥 Video Pipeline with Triton Inference and FastAPI UI

This project contains two Dockerized services:

1. **Triton Inference Server**: Serves the PyTorch model.
2. **FastAPI Video Pipeline**: UI + API to upload an `.mp4` video, send it for analysis, and return predictions using Triton.

---

## 📁 Project Structure
```bash 
├── model_repository/ # Triton model repository
│ └── exposurenet/
│ └── 1/
│ └── model.py # Python backend model
│ └── config.pbtxt
│
├── Video_pipeline/ # FastAPI video pipeline
│ ├── main.py # FastAPI server
│ ├── templates/
│ │ └── upload.html # Drag-and-drop upload UI
  ├── Videos/
│ │ └── *.mp4
│ ├── requirements.txt
│ └── Dockerfile.video # FastAPI Dockerfile
│
├── Dockerfile.torch # Triton Server Dockerfile
└── README.md
```
---

## 🚀 Service 1: Triton Inference Server

This service uses the [NVIDIA Triton Server](https://developer.nvidia.com/nvidia-triton-inference-server) with a PyTorch CPU backend.

### 🔧 Build Image

```bash
docker build -t tritonserver-torch -f Dockerfile.torch .
```

### ▶️ Run Triton Server
```bash
docker run -it --name triton_server --net=host -v $(pwd)/model_repository:/models tritonserver-torch tritonserver --model-repository=/models
```
--net=host: Required for communication from FastAPI

-v $(pwd)/model_repository:/models: Mount model repository into container

Triton will load the models from /models and serve inference requests.

---


## 🌐 Service 2: FastAPI Video Pipeline + UI

This FastAPI service provides:
  A UI (/) to upload .mp4 videos
  A backend (/analyze_video/) that:
  Saves the file
  Preprocesses if needed
  (Optionally) calls Triton Inference Server
  Returns predictions in the same HTML page

### 🔧 Build FastAPI Image

```bash 
cd Video_pipeline
docker build -t video-pipeline -f Dockerfile.video .
```

```bash 
docker run -p 5008:5008 --name video_pipeline video-pipeline
```

✨ Features
  📂 Drag & drop video upload UI

  📏 Max file size: 50MB (client & server side validated)

  ⚙️ Uses ffmpeg for preprocessing (installed in container)

  🧠 Connects to Triton Inference Server for model predictions

---