

## Project Description

The **Real-Time Air Drawing Application** is a touchless drawing system that allows users to draw on a virtual canvas using natural hand gestures in front of a webcam. The application provides a simple and natural way to interact with a computer without using a mouse, keyboard, or touchscreen.

The system uses **Python, OpenCV, MediaPipe, and NumPy** to capture webcam video, detect and track hand landmarks, recognize gestures, and convert hand movements into smooth drawing actions.

The project is designed for **college projects, laboratory demonstrations, computer vision learning, and viva presentations**. It also includes an **AI-ready command interface** that can be connected with an Agentic AI, LLM, or speech-recognition system to control canvas operations using natural-language commands.

### Project Guide

**Mr. Srikanth Dasari**

---

## Objective

The main objective of this project is to develop a simple and interactive **touchless human-computer interaction system** using hand gestures.

The application aims to:

- Enable drawing without physical contact with a computer.
- Detect and track hand movements in real time.
- Recognize different hand gestures for different actions.
- Provide smooth and natural drawing.
- Provide virtual controls for colors, eraser, clear, save, and brush size.
- Provide an AI-ready interface for future intelligent voice or natural-language control.

---

## Features

### 1. Touchless Air Drawing

Draw on a virtual canvas by moving your **index finger** in front of the webcam.

### 2. Natural Hand Gestures

| Gesture | Function |
|---|---|
| ☝️ Index finger UP | Draw Mode |
| ✌️ Index + Middle fingers UP | Selection / Control Mode |
| ✊ Closed Fist | Circular Eraser |
| 🖐️ Open / relaxed hand | Standby Mode |

### 3. Smooth Stroke Engine

- Exponential Moving Average (EMA) coordinate smoothing.
- Reduces hand tremors and tracking noise.
- Anti-aliased OpenCV line rendering.
- Rounded line caps for smooth strokes.
- Automatic stroke break when the hand leaves the camera frame.

### 4. Interactive On-Screen Toolbar

The application provides a virtual toolbar containing:

- 🔵 Blue
- 🟢 Green
- 🔴 Red
- ⚫ Black
- Eraser
- Clear Canvas
- Save Drawing
- Brush Size

The toolbar can be controlled using **two-finger selection mode**.

### 5. AI-Ready Architecture

The project includes `ai_interface.py`, which provides an interface for connecting an Agentic AI, LLM, or speech-recognition system.

For example, future versions can understand commands such as:

```text
"Change the color to blue"
"Clear the canvas"
"Increase brush size"
"Save the drawing"
```

### 6. Keyboard Failsafe

Keyboard shortcuts are also provided so that the application can be easily controlled during college demonstrations and presentations.

---

## Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- Computer Vision
- Hand Gesture Recognition
- Agentic AI-ready architecture

---

## Project Structure

```text
air_drawing/
│
├── main.py              # Main application loop and event handling
├── hand_tracking.py    # MediaPipe hand detection and landmark extraction
├── gestures.py         # Gesture classification and coordinate smoothing
├── drawing.py           # Virtual canvas, brushes, eraser and PNG export
├── ai_interface.py      # AI-ready natural language command dispatcher
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Installation and Setup

### Requirements

- Python 3.8 to 3.12+
- Working webcam
- Windows, macOS, or Linux

### Install Dependencies

Open a terminal or command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the main Python file:

```bash
python main.py
```

The webcam window will open automatically in mirror mode.

---

# How to Use

## 1. Gesture Guide

| Gesture | Pose | Action |
|---|---|---|
| ☝️ Draw | Only index finger raised | Draw smooth lines |
| ✌️ Select / Control | Index + middle fingers raised | Select toolbar buttons |
| ✊ Eraser | Closed fist | Erase strokes |
| 🖐️ Standby | Open or relaxed hand | Pause drawing |

---

## 2. Selecting Tools Using Gestures

1. Raise your **index and middle fingers**.
2. Move the pointer over a toolbar button.
3. Hold it over the button for approximately **0.35 seconds**.
4. A progress indicator appears.
5. The selected tool is automatically activated.

Available controls include:

```text
BLUE
GREEN
RED
BLACK
ERASER
CLEAR
SAVE
SIZE
```

---

## 3. Keyboard Shortcuts

| Key | Action |
|---|---|
| B | Blue brush |
| G | Green brush |
| R | Red brush |
| K | Black brush |
| E | Eraser |
| C | Clear canvas |
| S | Save drawing |
| + / = | Increase / cycle brush size |
| - | Reset to small brush |
| Q / ESC | Quit application |

Brush sizes:

```text
Small  → 4 px
Medium → 8 px
Large  → 16 px
```

---

# Saved Drawings

When the user selects **SAVE** or presses **S**, the drawing is saved as a PNG image in the `saved_drawings/` folder.

Example:

```text
saved_drawings/
└── air_drawing_20261027_161500.png
```

The saved drawing uses a clean white background.

---

# AI-Ready Command Interface

The project contains an `ai_interface.py` module that provides a command dispatcher for future AI integration.

It can be connected to an Agentic AI, Large Language Model, or speech-recognition system.

Example:

```python
from drawing import CanvasManager
from ai_interface import AICommandDispatcher

canvas = CanvasManager(640, 480)
ai = AICommandDispatcher(canvas)

ai.execute_command("Change the color to blue")
ai.execute_command("Increase brush size")
ai.execute_command("Clear the canvas")
ai.execute_command("Save the drawing")
```

This architecture allows the project to be extended from gesture-based control to **natural-language and voice-based interaction**.

---

# How the System Works

The basic workflow of the application is:

```text
Webcam
   ↓
Video Capture using OpenCV
   ↓
Hand Detection using MediaPipe
   ↓
21 Hand Landmark Detection
   ↓
Gesture Recognition
   ↓
Gesture Classification
   ↓
Drawing / Selection / Erasing
   ↓
Virtual Canvas
   ↓
Save Drawing
```

---

# Technical Implementation

## Hand Tracking

The application uses **MediaPipe** to detect and track the user's hand.

MediaPipe identifies **21 hand landmarks**, including:

- Wrist
- Finger joints
- Finger tips

These landmarks are used to determine the position and state of the fingers.

---

## Gesture Recognition

Finger positions are analyzed using the landmark coordinates.

For example, the application compares the **Y-coordinate of a fingertip** with its corresponding PIP joint.

Since the top of an image has a smaller Y-coordinate:

```text
Tip Y < PIP Y
```

indicates that the finger is raised.

Based on the detected fingers:

```text
Only Index Finger
        ↓
   Draw Mode

Index + Middle Finger
        ↓
 Selection Mode

All Fingers Folded
        ↓
  Eraser Mode
```

---

# Smooth Drawing

To make the drawing natural and smooth, the application uses **Exponential Moving Average (EMA)** smoothing.

The basic concept is:

```text
New Position =
α × Current Position
+
(1 - α) × Previous Position
```

This reduces small movements caused by hand tremors and camera noise.

The project also uses OpenCV's anti-aliased line rendering to produce smoother strokes.

---

# Dual-Buffer Canvas

The application uses a **dual-buffer architecture** consisting of a color layer and a dedicated stroke mask.

This is particularly useful for handling the **black brush** correctly.

Instead of treating black as an empty or transparent value, the system maintains the drawing information separately from the webcam frame.

This allows:

- True black strokes.
- Independent erasing.
- Clean rendering over the webcam feed.
- Better control of the virtual canvas.

---

# College Project Viva Questions

## Q1. How does hand tracking work?

**Answer:**  
The project uses Google's MediaPipe framework for hand tracking. MediaPipe detects the user's hand and identifies 21 important landmarks such as the wrist, finger joints, and fingertips. These landmarks are continuously tracked from the webcam video and are used for gesture recognition.

## Q2. How are gestures recognized?

**Answer:**  
The application compares the positions of finger tips and their corresponding joints. If the index finger is raised, the application enters drawing mode. If the index and middle fingers are raised, it enters selection mode. If the fingers are folded into a fist, the application activates the eraser.

## Q3. How are smooth lines generated?

**Answer:**  
The application uses EMA coordinate smoothing to reduce hand tremors and tracking noise. Consecutive smoothed coordinates are connected using OpenCV's anti-aliased line rendering to produce smooth and continuous strokes.

## Q4. Why is MediaPipe used?

**Answer:**  
MediaPipe provides real-time hand detection and tracking with 21 hand landmarks. It makes gesture recognition easier and allows the application to respond quickly to hand movements.

## Q5. Why is OpenCV used?

**Answer:**  
OpenCV is used for webcam capture, image processing, drawing lines, displaying the virtual canvas, handling keyboard input, and saving the final drawing.

## Q6. What is the purpose of the AI interface?

**Answer:**  
The AI interface provides an extension point for connecting an Agentic AI, LLM, or speech-recognition system. It can allow users to control drawing operations using natural-language commands such as "change the color to blue" or "clear the canvas."

---

# Future Enhancements

The project can be extended with:

- Voice-controlled drawing.
- Full Agentic AI integration.
- Natural-language commands.
- More advanced gesture recognition.
- Shape recognition.
- Automatic drawing correction.
- AI-based image generation from sketches.
- Multiple-hand support.
- Cloud storage for drawings.
- Web-based deployment.
- AI-assisted drawing suggestions.

---

# Project Guide

**Mr. Srikanth Dasari**

**B.Tech – Computer Science and Engineering**

This project demonstrates the practical application of **Computer Vision, Hand Gesture Recognition, Human-Computer Interaction, and AI-ready software architecture**.

---

## Conclusion

The **Real-Time Air Drawing Application** provides a simple and natural way to interact with a computer using hand gestures. By combining **OpenCV, MediaPipe, NumPy, gesture recognition, and an AI-ready architecture**, the system enables touchless drawing and interactive canvas control.

The project can be useful for **college demonstrations, classroom activities, presentations, computer vision experiments, and future AI-based human-computer interaction applications**.
