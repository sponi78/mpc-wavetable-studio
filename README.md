# MPC Wavetable Studio – Windows

Free desktop utility by **MPC Toolkit**  
https://mpctoolkit.com

## Main features

- Split a source WAV automatically using silence detection
- Silence controls use editable spin boxes with mouse arrows
- Automatically identify and deselect duplicate sounds
- Preview the complete source WAV
- Preview the currently selected sound
- Preview all enabled sounds in sequence
- Select a segment by clicking it in the waveform
- Drag the white start/end handles with the mouse
- Enter exact start/end positions in milliseconds
- Export a ready-to-copy MPC ZIP structure:

```
Oscillators/
  Wavetables/
    Wavetable1/
      format.json
      Wavetable1.wav
```

## Run from Python

```bat
py -m pip install -r requirements.txt
py mpc_wavetable_splitter.py
```

## Build the Windows application

Double-click:

- `build_windows.bat` for the recommended folder build
- `build_windows_onefile.bat` for a single EXE

The recommended build will be located at:

```
dist\MPC Wavetable Studio\MPC Wavetable Studio.exe
```

Distribute the complete `MPC Wavetable Studio` folder or package it with an installer.

## v1.3 compact/responsive layout

- Reduced vertical spacing and waveform height.
- Two-row toolbar for narrower windows.
- Export, Silence and Wavetable settings are organized in tabs.
- Segment table includes horizontal and vertical scrollbars.
- Selected-segment boundary controls remain visible above the footer.
- Minimum supported window size reduced to 820 x 600.
