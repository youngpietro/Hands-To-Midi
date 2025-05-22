# Hands-To-Midi
A simple python code that uses Mediapipe and OpenCv while tracking finger landmarks and translating them into chords (left fingers) and CC values (right fingers). 

---------------
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
  ---------------
