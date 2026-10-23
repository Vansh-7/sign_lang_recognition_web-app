# Architecture

`app.py` sets native thread limits before importing the application and calls
`asl_recognition.ui.main()`. The entry point remains stable for Community Cloud.

## Runtime flow

```text
Browser camera (up to 15 FPS)
    → WebRTC video worker
    → resize / mirror
    → MediaPipe Hands (one hand, at most 10 samples/second)
    → 63 float32 coordinates (21 × x/y/z)
    → shared NumPy dense model
    → confidence / stability / cooldown filtering
    → session-local sentence and annotated output frame

Streamlit sentence fragment (every 0.5 seconds)
    → locked processor snapshot
    → editable text / download / optional gTTS
```

## Ownership and cleanup

- `config.py` holds paths, class order, and resource limits. Paths resolve
  against the repository location, so changing the current directory is safe.
- `model.py` reads only the dense-layer weights from H5. It validates supported
  layers and dimensions. Dropout is inactive during inference. Weights remain
  read-only; inference does not load Keras, TensorFlow, or optimizer state.
- `services.py` caches one warmed model and inference lock per server process.
  It also caches temporary TURN configuration for one hour.
- `processor.py` owns one MediaPipe tracker per camera session, allocated on
  the first frame. A lock serializes frame processing, text edits, snapshots,
  and cleanup. `on_ended()` closes the tracker once. Async processing selects
  only the latest queued frame; recognition uses a monotonic clock.
- `ui.py` renders the page. Video workers do not call Streamlit APIs. The
  sentence tools use a timed fragment so a new letter does not reload the
  model, recreate the tracker, or rerun the entire page.

## Model contract

The bundled model accepts float32 batches of 63 coordinates. Its dense layers
are 63 → 128 (ReLU) → 64 (ReLU) → 28 (softmax), with two training-only dropout
layers. Output order is A–Z, `del`, `space`. Confidence must exceed 0.8.

Any replacement model must keep the feature order, label order, dimensions,
and supported activations. The loader deliberately rejects unsupported layer
types. Keep training changes separate until parity and camera tests pass.
The original NumPy replacement matched TensorFlow class predictions on all
512 sampled vectors; maximum probability difference was below 2.4e-7. This
is numerical parity evidence, not a recognition accuracy benchmark.

## Operational bounds

Frame processing width is capped at 480 pixels and recognition at 10 Hz.
Text is capped at 2,000 characters. Stale input frames are discarded; tracker
state is not shared across users. These are per-session work limits, not a
host-wide concurrency limit. Each additional session still uses CPU, memory,
and network resources.

Raw camera input travels to the app server, optionally through a TURN relay.
The application does not write frames or transcripts to disk. Text and audio
are held in session memory. gTTS receives text only when the user selects
**Speak Sentence**. External providers and hosting have their own retention
and operational policies.
