#!/usr/bin/env python
"""
Hand MIDI Controller with Adjustable Scaling, Debounce, Mode Toggles, Live Left-Hand Training,
Five Left-Hand Banks with Two-Hand Gesture-Based Bank Switching, and Speech Control for Bank Selection.

Bank 1: Chord trigger mode.
Banks 2-5: Toggle mode – each left-hand finger sends a CC toggle:
         • When its value rises (from below threshold) to 127, it toggles ON (sends CC value 127).
         • A second rising edge (after being active) toggles OFF (sends CC value 0).

Bank switching:
  • Keyboard: Press 'j' for next bank and 'k' for previous bank.
  • Two-hand gesture training (requires both hands on screen):
       - Press 'y' to train the “NEXT bank” gesture.
       - Press 'u' to train the “PREVIOUS bank” gesture.
       (An on‑screen counter shows training progress.)
  • Speech control:
       - Press 's' to toggle speech control.
       - When active, saying a command such as “bank one” or “bank two” sets the left‐hand bank accordingly.
  • Press 'i' to clear any trained gesture templates so that only keyboard (and speech) switching is active.

Other controls:
  • Press 't' to train left-hand chords.
  • Press 'm' and 'n' to toggle global and right-hand tracking modes.
  • Press 'l' to toggle hand landmarks.
  • Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import time
import json
import numpy as np
import mido
import speech_recognition as sr
from mido import Message
import tkinter as tk

# ----------------------------
# Left-hand Chord Training (live mode)
# ----------------------------
def train_left_hand_chords_live():
    print("\n=== Left Hand Chord Training ===")
    print("Available Chords:")
    chord_list = list(available_chords.items())
    for i, (chord_name, chord_info) in enumerate(chord_list, start=1):
        print(f"{i}. {chord_name}: {chord_info['description']}")
    mapping = {}
    for finger in ["thumb", "index", "middle", "pinky"]:
        choice = input(f"Select the chord number for left-hand {finger}: ")
        try:
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(chord_list):
                selected_chord = chord_list[choice_idx - 1][0]
                mapping[finger] = selected_chord
                print(f"Assigned {selected_chord} to {finger}.")
            else:
                print("Invalid number. Skipping this finger.")
        except ValueError:
            print("Invalid input, you need to input a number. Skipping this finger.")
    with open("left_hand_chord_mapping.json", "w") as f:
        json.dump(mapping, f)
    print("Left-hand chord mapping updated.\n")
    return mapping

# ----------------------------
# Available Chords Dictionary (20 chords)
# ----------------------------
available_chords = {
    "happy_chord": {"notes": [60, 64, 67], "description": "Bright and cheerful (C Major)"},
    "sad_chord": {"notes": [60, 63, 67], "description": "Melancholic and somber (C Minor)"},
    "angry_chord": {"notes": [60, 63, 66, 69], "description": "Intense and dissonant (Fully diminished seventh)"},
    "mysterious_chord": {"notes": [60, 64, 67, 70], "description": "Enigmatic and suspenseful"},
    "dreamy_chord": {"notes": [60, 65, 69], "description": "Soft and ambient (C Major add6)"},
    "tranquil_chord": {"notes": [62, 65, 69], "description": "Calm and soothing (D Minor)"},
    "tense_chord": {"notes": [60, 63, 66, 70], "description": "Unstable and tense"},
    "uplifting_chord": {"notes": [62, 66, 69], "description": "Inspiring and uplifting (D Major)"},
    "ominous_chord": {"notes": [60, 62, 67, 70], "description": "Dark and foreboding"},
    "reflective_chord": {"notes": [60, 64, 67, 72], "description": "Thoughtful and introspective"},
    "energetic_chord": {"notes": [62, 66, 69, 73], "description": "Vibrant and dynamic (D Major 7th)"},
    "mystical_chord": {"notes": [60, 65, 69, 74], "description": "Otherworldly and ethereal"},
    "blue_chord": {"notes": [60, 63, 67, 72], "description": "Bluesy and soulful"},
    "jazzy_chord": {"notes": [60, 64, 67, 70, 74], "description": "Smooth and complex (C6/9 style)"},
    "romantic_chord": {"notes": [60, 64, 67, 71], "description": "Warm and passionate (C Major 7th)"},
    "epic_chord": {"notes": [60, 64, 67, 72, 76], "description": "Grand and majestic"},
    "mellow_chord": {"notes": [60, 64, 67, 69], "description": "Smooth and mellow (C6)"},
    "suspense_chord": {"notes": [60, 65, 68], "description": "Suspenseful and unresolved"},
    "whimsical_chord": {"notes": [60, 63, 67, 70, 73], "description": "Playful and quirky"},
    "serene_chord": {"notes": [62, 65, 69, 74], "description": "Peaceful and serene (D Minor add6)"},
    "bb_major": {"notes": [58, 62, 65], "description": "B-flat Major chord"}
}

# ----------------------------
# Left Hand Default Mapping (Bank 1, chord mode)
# ----------------------------
default_left_hand_mapping = {
    "thumb": "bb_major",
    "index": "tranquil_chord",
    "middle": "dreamy_chord",
    "pinky": "serene_chord"
}
try:
    with open("left_hand_chord_mapping.json", "r") as f:
        left_hand_mapping = json.load(f)
except Exception as e:
    print(f"Could not load left_hand_chord_mapping.json, using default mapping. Error: {e}")
    left_hand_mapping = default_left_hand_mapping

# ----------------------------
# Right Hand Mapping and MIDI Settings
# ----------------------------
right_hand_mapping = {
    "thumb": {"landmark": 4, "cc": 1},
    "index": {"landmark": 8, "cc": 2},
    "middle": {"landmark": 12, "cc": 3},
    "pinky": {"landmark": 20, "cc": 4}
}
MIDI_PORT = "IAC Driver Bus 1"
LEFT_MIDI_CHANNEL = 0   # Left-hand messages (chords or CC toggles)
RIGHT_MIDI_CHANNEL = 1
LEFT_TRIGGER_THRESHOLD = 120

# ----------------------------
# Bank and Tracking Modality Settings
# ----------------------------
# left_bank_mode: 1 = Chord Trigger (default), 2-5 = Toggle CC modes.
left_bank_mode = 1

# Define CC mappings for banks 2-5 (each bank gets a unique set)
left_bank_cc_mappings = {
    2: {"thumb": 5, "index": 6, "middle": 7, "pinky": 8},
    3: {"thumb": 9, "index": 10, "middle": 11, "pinky": 12},
    4: {"thumb": 13, "index": 14, "middle": 15, "pinky": 16},
    5: {"thumb": 17, "index": 18, "middle": 19, "pinky": 20}
}

# For banks 2-5, maintain a toggle state per finger.
left_bank_states = {
    2: {"thumb": False, "index": False, "middle": False, "pinky": False},
    3: {"thumb": False, "index": False, "middle": False, "pinky": False},
    4: {"thumb": False, "index": False, "middle": False, "pinky": False},
    5: {"thumb": False, "index": False, "middle": False, "pinky": False}
}
# And store the last measured value for each finger (for rising edge detection).
left_bank_last_values = {
    2: {"thumb": 0, "index": 0, "middle": 0, "pinky": 0},
    3: {"thumb": 0, "index": 0, "middle": 0, "pinky": 0},
    4: {"thumb": 0, "index": 0, "middle": 0, "pinky": 0},
    5: {"thumb": 0, "index": 0, "middle": 0, "pinky": 0}
}

tracking_mode = 1       # Global (left-hand) mode: 1 = Global Y-axis, 2 = Relative-to-center.
right_tracking_mode = 1 # Right-hand mode: 1 = Global Y-axis, 2 = Relative-to-center.
MAX_EXTENSION = 0.35
LEFT_MAX_EXTENSION = 0.25
LEFT_MAX_EXTENSION_THUMB = 0.2
LEFT_MAX_EXTENSION_PINKY = 0.2
RIGHT_SCALE_FACTOR = 1.2
landmarks_visible = True

# ----------------------------
# Debounce and Hand Appearance Settings
# ----------------------------
FIRST_TRIGGER_DELAY = 1.0        # Wait at least 1.0 sec after chord stop before triggering new chord.
TRIGGER_DEBOUNCE_TIME = 0.3        # Minimum time between successive triggers.
right_hand_trigger_delay = 1.0     # Wait 1 sec after right hand appears before processing CC.
left_hand_trigger_delay = 1.0      # Wait 1 sec after left hand appears before processing.
last_chord_stop_time = 0
last_chord_trigger_time = 0
right_hand_was_absent = True
right_hand_reappearance_time = 0
left_hand_was_absent = True
left_hand_reappearance_time = 0

# ----------------------------
# Two-Hand Gesture Training for Bank Switching
# ----------------------------
# Templates for bank switch commands (combined left+right).
next_gesture_template = None   # For switching to NEXT bank.
prev_gesture_template = None   # For switching to PREVIOUS bank.
# Flags to avoid repeated triggers.
next_gesture_active = False
prev_gesture_active = False
# Training mode flags and buffers.
training_next = False    # When True, collect samples for NEXT bank gesture.
training_prev = False    # When True, collect samples for PREVIOUS bank gesture.
SAMPLES_NEEDED = 30
next_gesture_samples = []
prev_gesture_samples = []
# Threshold for gesture matching (in normalized coordinates)
GESTURE_THRESHOLD = 0.03

def compute_average_distance(template, current):
    """Compute average Euclidean distance between corresponding landmarks.
       Both template and current are lists of (x, y) tuples.
    """
    if len(template) != len(current):
        return float('inf')
    distances = [np.sqrt((tx - cx)**2 + (ty - cy)**2) for (tx, ty), (cx, cy) in zip(template, current)]
    return sum(distances) / len(distances)

# ----------------------------
# Speech Recognition Global Variables and Callback
# ----------------------------
speech_control_active = False
speech_stop_listening = None

def speech_callback(recognizer, audio):
    global left_bank_mode
    try:
        text = recognizer.recognize_google(audio)
        text = text.lower()
        print(f"[Speech] Recognized: {text}")
        if "bank" in text:
            mapping_numbers = {"one": 1, "1": 1,
                               "two": 2, "2": 2,
                               "three": 3, "3": 3,
                               "four": 4, "4": 4,
                               "five": 5, "5": 5}
            for word, num in mapping_numbers.items():
                if word in text:
                    left_bank_mode = num
                    print(f"Switched to left-hand bank {num} (speech)")
                    break
    except Exception as e:
        # If recognition fails, ignore silently or print error.
        # print(f"[Speech] Error: {e}")
        pass

# ----------------------------
# Helper Function: Compute Hand Center
# ----------------------------
def compute_center(landmarks):
    indices = [0, 5, 9, 13, 17]
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices if i < len(landmarks)])
    if pts.size == 0:
        return None
    return np.mean(pts, axis=0)

# ----------------------------
# Open MIDI Output
# ----------------------------
try:
    midi_out = mido.open_output(MIDI_PORT)
    print(f"Opened MIDI output on {MIDI_PORT}")
except Exception as e:
    print(f"Error opening MIDI port '{MIDI_PORT}': {e}")
    exit(1)

# ----------------------------
# Initialize MediaPipe Hands and Video Capture
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# ----------------------------
# Chord State (Bank 1)
# ----------------------------
current_chord = None
active_chord_notes = []

def trigger_chord(new_chord_name):
    global current_chord, active_chord_notes, last_chord_stop_time, last_chord_trigger_time
    if new_chord_name == current_chord:
        return
    if active_chord_notes:
        for note in active_chord_notes:
            midi_out.send(Message('note_off', channel=LEFT_MIDI_CHANNEL, note=note, velocity=64))
        active_chord_notes.clear()
        current_chord = None
        last_chord_stop_time = time.time()
    if new_chord_name is None:
        return
    chord_info = available_chords.get(new_chord_name)
    if chord_info:
        for note in chord_info["notes"]:
            midi_out.send(Message('note_on', channel=LEFT_MIDI_CHANNEL, note=note, velocity=100))
        active_chord_notes.extend(chord_info["notes"])
        current_chord = new_chord_name
        last_chord_trigger_time = time.time()
        print(f"Triggered chord: {new_chord_name} -> {chord_info['notes']}")
    else:
        print(f"Chord {new_chord_name} not found.")

print("=== Hand MIDI Controller ===")
print("Controls:")
print("  'q' - quit")
print("  'm' - toggle global (left-hand) tracking mode")
print("  'n' - toggle right-hand tracking mode")
print("  'l' - toggle landmarks display")
print("  't' - left-hand chord training")
print("  'j' - next bank (keyboard)")
print("  'k' - previous bank (keyboard)")
print("  'y' - train NEXT bank gesture (requires BOTH hands)")
print("  'u' - train PREV bank gesture (requires BOTH hands)")
print("  'i' - clear trained gestures (then only j/k & speech work)")
print("  's' - toggle speech control for bank switching")

# Left-hand finger landmarks used for drawing and processing.
left_fingers = {"thumb": 4, "index": 8, "middle": 12, "pinky": 20}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        tracking_mode = 2 if tracking_mode == 1 else 1
        print(f"Global mode: {'Relative-to-center' if tracking_mode==2 else 'Global Y-axis'}")
        time.sleep(0.2)
    elif key == ord('n'):
        right_tracking_mode = 2 if right_tracking_mode == 1 else 1
        print(f"Right-hand mode: {'Relative-to-center' if right_tracking_mode==2 else 'Global Y-axis'}")
        time.sleep(0.2)
    elif key == ord('l'):
        landmarks_visible = not landmarks_visible
        print(f"Landmarks display {'enabled' if landmarks_visible else 'disabled'}")
        time.sleep(0.2)
    elif key == ord('t'):
        print("Entering left-hand chord training mode...")
        new_mapping = train_left_hand_chords_live()
        if new_mapping:
            left_hand_mapping = new_mapping
        time.sleep(0.5)
    elif key == ord('j'):
        left_bank_mode += 1
        if left_bank_mode > 5:
            left_bank_mode = 1
        print(f"Switched to left-hand bank {left_bank_mode} (keyboard next)")
        time.sleep(0.2)
    elif key == ord('k'):
        left_bank_mode -= 1
        if left_bank_mode < 1:
            left_bank_mode = 5
        print(f"Switched to left-hand bank {left_bank_mode} (keyboard previous)")
        time.sleep(0.2)
    elif key == ord('y'):
        training_next = True
        next_gesture_samples = []
        print("Training for NEXT bank gesture: Please hold BOTH hands steady in view.")
        time.sleep(0.2)
    elif key == ord('u'):
        training_prev = True
        prev_gesture_samples = []
        print("Training for PREVIOUS bank gesture: Please hold BOTH hands steady in view.")
        time.sleep(0.2)
    elif key == ord('i'):
        next_gesture_template = None
        prev_gesture_template = None
        print("Gesture templates cleared. Now only keyboard and speech switching are available.")
        time.sleep(0.2)
    elif key == ord('s'):
        # Toggle speech control
        if not speech_control_active:
            r = sr.Recognizer()
            m = sr.Microphone()
            speech_stop_listening = r.listen_in_background(m, speech_callback)
            speech_control_active = True
            print("Speech control activated.")
        else:
            if speech_stop_listening:
                speech_stop_listening(wait_for_stop=False)
            speech_control_active = False
            print("Speech control deactivated.")
        time.sleep(0.2)

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_time = time.time()
    frame_height, frame_width = frame.shape[:2]

    right_controls = {}
    left_values = {}
    detected_hands = []  # List of tuples: (hand_coords, label)
    right_hand_detected = False
    left_hand_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            landmarks = hand_landmarks.landmark
            # Get full set of normalized (x, y) coordinates.
            hand_coords = [(lm.x, lm.y) for lm in landmarks]
            detected_hands.append((hand_coords, label))
            if landmarks_visible:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if label == "Right":
                right_hand_detected = True
                if right_tracking_mode == 1:
                    def get_value_right(lm_index):
                        return int((1 - landmarks[lm_index].y) * 127) if lm_index < len(landmarks) else 0
                else:
                    center_right = compute_center(landmarks)
                    def get_value_right(lm_index):
                        if lm_index >= len(landmarks) or center_right is None:
                            return 0
                        pt = np.array([landmarks[lm_index].x, landmarks[lm_index].y])
                        norm = min(np.linalg.norm(pt - center_right) / MAX_EXTENSION, 1.0)
                        return int(norm * 127)
                for finger_name, mapping in right_hand_mapping.items():
                    lm_index = mapping["landmark"]
                    value = get_value_right(lm_index)
                    value = int(value * RIGHT_SCALE_FACTOR)
                    value = max(0, min(127, value))
                    right_controls[finger_name] = value
                if right_hand_was_absent:
                    right_hand_reappearance_time = current_time
                    right_hand_was_absent = False
                if (current_time - right_hand_reappearance_time) >= right_hand_trigger_delay:
                    rx = frame_width - 200
                    ry = 30
                    for i, finger_name in enumerate(right_hand_mapping.keys()):
                        cc_value = right_controls.get(finger_name, 0)
                        midi_out.send(Message('control_change', channel=RIGHT_MIDI_CHANNEL,
                                                control=right_hand_mapping[finger_name]["cc"], value=cc_value))
                        cv2.putText(frame, f"Right {finger_name}: {cc_value}", (rx, ry + i*30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif label == "Left":
                left_hand_detected = True
                if tracking_mode == 1:
                    def get_value_left(lm_index, finger_name=None):
                        return int((1 - landmarks[lm_index].y) * 127) if lm_index < len(landmarks) else 0
                else:
                    center_left = compute_center(landmarks)
                    def get_value_left(lm_index, finger_name=None):
                        if lm_index >= len(landmarks) or center_left is None:
                            return 0
                        pt = np.array([landmarks[lm_index].x, landmarks[lm_index].y])
                        max_ext = LEFT_MAX_EXTENSION_THUMB if finger_name=="thumb" else (LEFT_MAX_EXTENSION_PINKY if finger_name=="pinky" else LEFT_MAX_EXTENSION)
                        norm = min(np.linalg.norm(pt - center_left) / max_ext, 1.0)
                        return int(norm * 127)
                temp_left_values = {}
                for finger_name, lm_index in left_fingers.items():
                    temp_left_values[finger_name] = get_value_left(lm_index, finger_name=finger_name)
                left_values = temp_left_values.copy()
    else:
        right_hand_was_absent = True
        left_hand_was_absent = True

    # ----------------------------
    # Two-Hand Gesture Training for Bank Switching
    # ----------------------------
    # Only collect training samples if BOTH hands are visible.
    left_sample = None
    right_sample = None
    for coords, label in detected_hands:
        if label == "Left" and left_sample is None:
            left_sample = coords
        if label == "Right" and right_sample is None:
            right_sample = coords
    if training_next and left_sample is not None and right_sample is not None:
        combined = left_sample + right_sample
        next_gesture_samples.append(combined)
        cv2.putText(frame, f"Training NEXT Gesture: {len(next_gesture_samples)}/{SAMPLES_NEEDED}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(next_gesture_samples) >= SAMPLES_NEEDED:
            arr = np.array(next_gesture_samples)
            next_gesture_template = np.mean(arr, axis=0).tolist()
            training_next = False
            next_gesture_samples = []
            print("NEXT bank gesture trained.")
    if training_prev and left_sample is not None and right_sample is not None:
        combined = left_sample + right_sample
        prev_gesture_samples.append(combined)
        cv2.putText(frame, f"Training PREV Gesture: {len(prev_gesture_samples)}/{SAMPLES_NEEDED}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(prev_gesture_samples) >= SAMPLES_NEEDED:
            arr = np.array(prev_gesture_samples)
            prev_gesture_template = np.mean(arr, axis=0).tolist()
            training_prev = False
            prev_gesture_samples = []
            print("PREVIOUS bank gesture trained.")

    # ----------------------------
    # Evaluate Two-Hand Gestures for Bank Switching
    # ----------------------------
    if left_sample is not None and right_sample is not None:
        combined_current = left_sample + right_sample
        if next_gesture_template is not None:
            avg_dist = compute_average_distance(next_gesture_template, combined_current)
            if avg_dist < GESTURE_THRESHOLD:
                if not next_gesture_active:
                    left_bank_mode += 1
                    if left_bank_mode > 5:
                        left_bank_mode = 1
                    print(f"Switched to left-hand bank {left_bank_mode} (NEXT gesture)")
                    next_gesture_active = True
            else:
                next_gesture_active = False
        if prev_gesture_template is not None:
            avg_dist = compute_average_distance(prev_gesture_template, combined_current)
            if avg_dist < GESTURE_THRESHOLD:
                if not prev_gesture_active:
                    left_bank_mode -= 1
                    if left_bank_mode < 1:
                        left_bank_mode = 5
                    print(f"Switched to left-hand bank {left_bank_mode} (PREV gesture)")
                    prev_gesture_active = True
            else:
                prev_gesture_active = False

    # ----------------------------
    # Left-hand Appearance (Drawing)
    # ----------------------------
    if left_hand_detected:
        if left_hand_was_absent:
            left_hand_reappearance_time = current_time
            left_hand_was_absent = False
        if (current_time - left_hand_reappearance_time) < left_hand_trigger_delay:
            cv2.putText(frame, "Left hand warming up...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            left_values = {}
        else:
            lx = 10
            ly = 30
            for i, finger_name in enumerate(left_fingers.keys()):
                cv2.putText(frame, f"Left {finger_name}: {left_values.get(finger_name, 0)}", (lx, ly + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        left_hand_was_absent = True

    # ----------------------------
    # Left Hand Processing (Branch by Bank Mode)
    # ----------------------------
    if left_values:
        if left_bank_mode == 1:
            # Bank 1: Chord Trigger Mode.
            if len(left_values) == 4 and all(val >= 125 for val in left_values.values()):
                if current_chord is not None:
                    trigger_chord(None)
                    last_chord_stop_time = current_time
            else:
                triggered_chord = None
                for finger_name, val in left_values.items():
                    if val >= LEFT_TRIGGER_THRESHOLD:
                        mapped_chord = left_hand_mapping.get(finger_name)
                        if mapped_chord:
                            triggered_chord = mapped_chord
                            break
                if triggered_chord:
                    if current_chord is None:
                        if (current_time - last_chord_stop_time) >= FIRST_TRIGGER_DELAY:
                            if (current_time - last_chord_trigger_time) >= TRIGGER_DEBOUNCE_TIME:
                                trigger_chord(triggered_chord)
                    else:
                        if (current_time - last_chord_trigger_time) >= TRIGGER_DEBOUNCE_TIME:
                            trigger_chord(triggered_chord)
        else:
            # Banks 2-5: Toggle CC Mode.
            for finger_name in left_fingers.keys():
                current_val = left_values.get(finger_name, 0)
                last_val = left_bank_last_values[left_bank_mode][finger_name]
                if last_val < 127 and current_val >= 125:
                    if not left_bank_states[left_bank_mode][finger_name]:
                        midi_out.send(Message('control_change', channel=LEFT_MIDI_CHANNEL,
                                                control=left_bank_cc_mappings[left_bank_mode][finger_name], value=127))
                        left_bank_states[left_bank_mode][finger_name] = True
                        print(f"Bank {left_bank_mode}: {finger_name} toggled ON (CC {left_bank_cc_mappings[left_bank_mode][finger_name]}:127)")
                    else:
                        midi_out.send(Message('control_change', channel=LEFT_MIDI_CHANNEL,
                                                control=left_bank_cc_mappings[left_bank_mode][finger_name], value=0))
                        left_bank_states[left_bank_mode][finger_name] = False
                        print(f"Bank {left_bank_mode}: {finger_name} toggled OFF (CC {left_bank_cc_mappings[left_bank_mode][finger_name]}:0)")
                left_bank_last_values[left_bank_mode][finger_name] = current_val

    # ----------------------------
    # Display Modalities (bottom right)
    # ----------------------------
    modality_text = f"Global: {('Global Y' if tracking_mode==1 else 'Relative-to-center')} | Right: {('Global Y' if right_tracking_mode==1 else 'Relative-to-center')}"
    bank_text = f"Left Bank: {left_bank_mode}"
    speech_text = f"Speech: {'On' if speech_control_active else 'Off'}"
    combined_text = modality_text + " | " + bank_text + " | " + speech_text
    (mod_w, _), _ = cv2.getTextSize(combined_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    modality_x = frame_width - mod_w - 10
    modality_y = frame_height - 10
    cv2.putText(frame, combined_text, (modality_x, modality_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # ----------------------------
    # Display Currently Playing Chord (bottom left)
    # ----------------------------
    if current_chord is not None:
        cv2.putText(frame, f"Playing: {current_chord}", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No chord playing", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hands to MIDI Controller", frame)

cap.release()
cv2.destroyAllWindows()