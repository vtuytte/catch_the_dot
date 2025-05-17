# Catch the Dot 

**Catch the Dot** is a small interactive game where you use your **index finger** to touch red dots on your screen — all detected through your **webcam**. It uses **OpenCV** and **MediaPipe** for finger tracking and gameplay interaction.

---

## How to Play

You can start the game using the `main.py` script with one of two modes:

### Singleplayer
```bash
python main.py singleplayer
```
Catch as many red dots as you can using one index finger.

### Multiplayer
```bash
python main.py multiplayer
```
Compete between your **left and right hands**. Each hand tries to catch more dots than the other.

---

## Code Overview

- **`source.py`**
  - `index_detector` class: Handles index finger detection.
  - `red_dot` class: Controls red dot behavior and logic.

- **`config.py`**
  - Contains configuration parameters for the webcam and game window.

- **`game.py`**
  - `run_singleplayer()`: Starts the singleplayer game.
  - `run_multiplayer()`: Starts the multiplayer game.

---

## Requirements

Made using mediapipe version 0.10.21.

```bash
pip install "mediapipe>=0.10.21"
```


---

## Notes

- A webcam is required to play.

---


