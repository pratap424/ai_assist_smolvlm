
 🚀 An AI Powered Vision and Navigation Assistance System

A modular, real time deep learning system that transitions from basic scene understanding to depth aware intelligent navigation, integrating object detection, depth estimation, vision language models, and audio visual feedback to assist in perceiving, interpreting, and interacting with the environment.

   

 📄 1. Abstract / Invention Overview

This invention presents a multi phase, AI driven system designed for comprehensive visual environment understanding and intelligent navigation assistance. It synergistically combines real time object detection, scene summarization, depth estimation, and navigation planning to deliver both descriptive and actionable feedback.

Key Capabilities:
  Detect and describe what's seen in an image or video
  Answer questions about a visual scene
  Estimate spatial layout from 2D input
  Plan a navigable path and guide users via instructions

   

 🎯 2. Problem Statement & Background

Existing visual assistive tools offer limited, disconnected features. Most fail to:
  Integrate spatial understanding with scene interpretation
  Offer navigation guidance using real time environmental data

 ❗This project solves that by:
  Combining visual recognition + spatial modeling
  Supporting both passive understanding (captions, Q&A) and active assistance (pathfinding, guidance)
  Building a modular and extensible system for various applications

   

 🔧 3. System Overview

The system is built in two evolutionary phases, each enhancing capabilities:

   

 ✅ Phase 1(Earlier One) – Foundational Visual Intelligence

| Feature                     | Description                                                                 |
|                            |                                                                             |
| 🟨 Object Detection         | Uses `YOLOv8n` for identifying objects with bounding boxes & confidence     |
| 🖼️ Scene Description        | `Salesforce BLIP` generates captions for images                             |
| 🎞️ Video Summarization     | BLIP processes keyframes and generates a coherent summary                   |
| ❓ Visual Question Answering| BLIP VQA answers natural language questions about a visual input            |
| 🖱️ Interactive GUI         | `Tkinter` based GUI for loading/capturing media and showing outputs         |

   

 ✅ Phase 2(This one) – Advanced Spatial Awareness & Navigation

| Feature                      | Description                                                                 |
|                             |                                                                             |
| 🔍 Upgraded Object Detection | `YOLOv8l` for higher accuracy and better feature extraction                 |
| 🌊 Depth Estimation         | `MiDaS DPT_Large` estimates distance to scene elements from single image    |
| 🗺️ 2D Environment Grid      | Converts camera + depth input into a grid with free space and obstacles     |
| 🧭 Pathfinding               | `A Algorithm` computes shortest path to chosen object                     |
| 🔊 Audio Feedback            | Uses `pyttsx3` for offline TTS instructions                                |
| 🖼️ SmolVLM Scene Summary    | `SmolVLM2` generates high quality image/video descriptions and instructions |
| 📟 Navigation GUI           | Displays real time feed, depth map, grid with planned path, and feedback   |

   

 📂 4. Codebase Breakdown



.
├── smolvlm\_module.py        Scene/video captioning, navigation via SmolVLM
├── yolo\_module.py           YOLOv8 object detection with bbox + confidence
├── midas\_module.py          Monocular depth estimation (MiDaS)
├── qna\_module.py            Context aware QnA with Sentence Transformers

`

   

 🔍 `yolo_module.py`
  Uses `ultralytics/yolov8l.pt`
  Outputs object label, bounding box, and confidence
python
detector = YOLODetector()
detections = detector.detect("img.jpg")
`

   

 🧠 `smolvlm_module.py`

 Describes images/videos using SmolVLM2
 Plans natural language navigation using labeled detections and positions

python
describe_image("img.jpg")
describe_video("video.mp4")
plan_navigation(detections, target_index=0)


   

 🌊 `midas_module.py`

 Estimates depth using `MiDaS DPT_Large`
 Returns numeric depth map and grayscale visualization

python
depth_model = MiDaSDepth()
depth_map, depth_img = depth_model.estimate_depth(pil_image)


   

 ❓ `qna_module.py`

 Uses `SentenceTransformer (MiniLM)` for contextual QnA
 Accepts any scene description and allows relevant Q\&A

python
qna = QnASystem()
qna.update_context("A man is standing near a car...")
qna.answer_question("What is the man doing?")


   

 🛠️ 5. Technologies & Methodologies

| Category                   | Tools / Models                                                         |
|                            |                                                                        |
| 👁️ Object Detection       | YOLOv8n (Phase 1), YOLOv8l (Phase 2)                                   |
| 🖼️ Vision Language Models | Salesforce BLIP (Captioning, VQA), SmolVLM2 (Image/Video + Navigation) |
| 🌊 Depth Estimation        | MiDaS DPT\_Large (from Intel ISL)                                      |
| 🔎 Pathfinding             | A\ Search Algorithm                                                   |
| 🛠️ Programming Stack      | Python, PyTorch, Hugging Face, OpenCV, Tkinter, PIL, NumPy             |
| 🔊 TTS                     | pyttsx3 for offline audio output                                       |
| ❓ QnA Embeddings           | SentenceTransformers (all MiniLM L6 v2)                                |

   

 💡 6. Potential Applications

 🧑‍🦯 Assistive navigation for visually impaired individuals
 🚁 Autonomous drones with spatial awareness
 🧠 Vision language research in robotics
 🧭 Smart surveillance with scene understanding
 🎓 Educational tools using QnA on visual data
 📱 Augmented reality systems for enhanced perception

   

 🏁 Key Achievements

 ✔️ Full stack AI integration: detection + VLMs + navigation
 ✔️ Real time depth aware navigation with pathfinding
 ✔️ GUI based interface for interaction and control
 ✔️ Modular, expandable architecture for future upgrades

   

 📥 Installation

bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install transformers
pip install sentence transformers
pip install opencv python
pip install decord
pip install pyttsx3

https://github.com/user-attachments/assets/51f320ad-c562-40ae-b757-6674f3705a12



