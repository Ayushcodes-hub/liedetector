# 👁️‍🗨️ Multimodal Deception Assessment Engine (MDAE)

---

## 1. Executive Synopsis
The **Multimodal Deception Assessment Engine (MDAE)** is an advanced, real-time diagnostic architecture engineered to evaluate human veracity through an amalgamated synthesis of behavioral biometrics and lexical content extraction. 

By seamlessly interlacing high-frequency computer vision, transient facial-landmark analytics, and asynchronous vocal-acoustic transcription, this engine scrutinizes micro-expressions—specifically, **ocular palpitation dynamics**—alongside **cognitive evasion markers** within a unified, interactive graphical interface.

---

## 2. Theoretical Architecture & Methodology

The application establishes a tripartite diagnostic pipeline to deduce cognitive dissonance and physiological anomalies often correlated with premeditated dissimulation:

### A. Computer Vision & Ocular Morphometry
Utilizing a convolutional topology via the **MediaPipe Face Mesh** pipeline, the engine tracks subtle structural variations around the periorbital geography. The primary metric computed is the **Eye Aspect Ratio (EAR)**, a geometric formulation mapped across specialized coordinate arrays ($left\_upper$, $left\_lower$, $right\_upper$, $right\_lower$). 

$$\text{EAR} = \frac{\|\mathbf{p}_2 - \mathbf{p}_6\| + \|\mathbf{p}_3 - \mathbf{p}_5\|}{2\|\mathbf{p}_1 - \mathbf{p}_4\|}$$

When the metric falls beneath a calibrated empirical threshold ($\text{EAR} < 0.22$), a baseline transient event is cached. Upon restoration of the ratio, a deliberate blink is cataloged.

### B. Temporal Behavioral Metrics
Human conversation typically exhibits a stable baseline of ocular oscillations (roughly 15–25 blinks per minute). The engine computes a rolling temporal derivative of this frequency:

$$\text{Blinks per Minute (BPM)} = \left(\frac{\text{Blink Count}}{\Delta t}\right) \times 60$$

Anomalous physiological states are flagged when the subject exhibits either **ocular fixation** ($\text{BPM} < 8$), suggestive of intense cognitive load and hyper-focus, or **paroxysmal fluttering** ($\text{BPM} > 35$), denoting acute autonomic nervous system arousal.

### C. Lexical Evasion Taxonomy
Concurrently, an asynchronous ambient acoustic thread ingests spoken waveforms, translating them via a semantic parser into normalized textual strings. This transcription is passed through a deterministic string-matching sieve optimized to detect linguistic obfuscation profiles, linguistic hedging, and explicit truth-assertions (e.g., *"honestly"*, *"to be honest"*, *"maybe"*).

---

## 3. Dependency Matrix

To deploy the engine, ensure your execution environment contains the following prerequisite binaries and libraries:

| Dependency | Purpose |
| :--- | :--- |
| `Python 3.10+` | Core execution runtime. |
| `OpenCV` | Hardware frame acquisition, color-space inversion ($BGR \rightarrow RGB$), and matrix transformations. |
| `MediaPipe` | High-fidelity facial landmark topology and machine learning inference pipeline. |
| `SpeechRecognition` | Asynchronous audio ingestion, ambient-noise mitigation, and Google Web Speech API orchestration. |
| `Pillow (PIL)` | Raster image serialization and transformation for Tkinter GUI compatibility. |
| `PyAudio` | Crucial native I/O audio binding required by `SpeechRecognition` to interface with hardware microphones. |

---

## 4. Operational Guide

### Installation of Dependencies
Execute the following consolidation command within your terminal deployment environment:

```bash
pip install opencv-python mediapipe SpeechRecognition Pillow pyaudio
Script Execution
Initialize the main execution thread by running the application script:

Bash
python lie_detector.pyUser Interface Interaction Protocol
Calibration Phase: Upon initialization, the system actively samples ambient acoustic noise for 1,000 milliseconds to establish a noise-floor baseline. Ensure your environment remains acoustic-neutral during this phase.

Interrogation Ingestion: The interface presents a prompt: "QUESTION: Did you eat the last slice of cake?"

Subject Response: The subject must position their face within the camera viewport and state their answer aloud.

Automated Synthesis: The text transcription box will dynamically display the parsed phrase while the core decision framework instantly shifts the Verdict Panel state to display the calculated diagnostic output based on behavioral-lexical alignment.

5. Algorithmic Decision Flow
              [ Spoken Answer Ingested ]
                          |
             +------------+------------+
             |                         |
    [ Parse Textual Data ]    [ Compute Ocular Frequency ]
             |                         |
     Categorize Voice:                 |
  (Says No / Says Yes / Evasive)       |
             |                         |
             +------------+------------+
                          |
            Is Blinking Anomalous (<8 or >35 BPM) 
               OR Contains Evasion Tokens?
                          |
                +---------+---------+
                |                   |
               YES                  NO
                |                   |
      [ 🚨 Deception Flagged ]   [ ✅ Veracity Validated ]
6. Salient Architectural Disclaimers
[!WARNING]
This software utilizes standard heuristic models correlating physiological stress markers with vocal cues. It is intended strictly for computational experimentation, academic analysis, and novelty entertainment. It does not replace forensic psychophysiological polygraph assessments.
