import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from asl_recognition import processor as video
from asl_recognition import ui
from asl_recognition.config import MAX_SENTENCE_LENGTH, MODEL_PATH, PROJECT_ROOT
from asl_recognition.model import ASLModel


@pytest.fixture
def processor(monkeypatch):
    tracker = Mock()
    tracker.process.return_value = SimpleNamespace(multi_hand_landmarks=None)
    monkeypatch.setattr(video.mp.solutions.hands, "Hands", Mock(return_value=tracker))
    instance = video.ASLProcessor(Mock(), threading.Lock())
    yield instance, tracker
    instance.on_ended()


def test_video_work_is_throttled_and_downscaled(processor, monkeypatch):
    instance, tracker = processor
    clock = iter([0, 0.02, 0.05, 0.11])
    monkeypatch.setattr(video.time, "monotonic", lambda: next(clock))
    frame = video.av.VideoFrame.from_ndarray(np.zeros((720, 1280, 3), np.uint8), format="bgr24")
    for _ in range(4):
        output = instance.recv(frame)
        assert output.width == 480
        assert output.height == 270
    assert tracker.process.call_count == 2
    assert tracker.process.call_args.args[0].shape == (270, 480, 3)
    assert instance.snapshot()[1] is None


def test_stop_releases_tracker_once_and_rejects_more_work(processor):
    instance, tracker = processor
    frame = video.av.VideoFrame.from_ndarray(np.zeros((360, 480, 3), np.uint8), format="bgr24")
    instance.recv(frame)
    instance.on_ended()
    instance.on_ended()
    assert instance.recv(frame) is frame
    tracker.close.assert_called_once()
    assert tracker.process.call_count == 1


def test_queued_frames_only_process_newest(processor):
    instance, _ = processor
    instance.recv = Mock(return_value="processed")
    assert asyncio.run(instance.recv_queued(["stale", "newest"])) == ["processed"]
    instance.recv.assert_called_once_with("newest")


def test_uncertain_frames_break_stability_and_clear_resets(processor):
    instance, _ = processor
    confident = np.zeros(28, np.float32)
    confident[0] = 0.99
    for _ in range(4):
        instance._accept_prediction(confident)
    instance._accept_prediction(np.full(28, 1 / 28))
    instance._accept_prediction(confident)
    assert instance.snapshot()[0] == ""
    for _ in range(4):
        instance._accept_prediction(confident)
    assert instance.snapshot()[0] == "A"
    instance.set_sentence("")
    assert instance.snapshot()[0] == ""
    assert instance.consecutive_frames == instance.cooldown == 0


def test_sessions_do_not_share_trackers_or_text(monkeypatch):
    first_tracker, second_tracker = Mock(), Mock()
    for tracker in (first_tracker, second_tracker):
        tracker.process.return_value = SimpleNamespace(multi_hand_landmarks=None)
    monkeypatch.setattr(
        video.mp.solutions.hands, "Hands", Mock(side_effect=[first_tracker, second_tracker])
    )
    first = video.ASLProcessor(Mock(), threading.Lock(), "first")
    second = video.ASLProcessor(Mock(), threading.Lock(), "second")
    frame = video.av.VideoFrame.from_ndarray(np.zeros((100, 100, 3), np.uint8), format="bgr24")
    first.recv(frame)
    second.recv(frame)
    first.on_ended()
    second_tracker.close.assert_not_called()
    assert second.snapshot()[0] == "second"
    second.on_ended()


def test_sentence_is_bounded(processor):
    instance, _ = processor
    instance.set_sentence("A" * 10000)
    assert len(instance.snapshot()[0]) == MAX_SENTENCE_LENGTH


def test_saved_model_returns_finite_class_probabilities():
    model = ASLModel(MODEL_PATH)
    output = model(np.zeros((1, 63), dtype=np.float32))
    assert output.shape == (1, 28)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError):
        model(np.zeros((1, 62)))


def test_sentence_tools_sync_edit_and_clear(monkeypatch):
    from streamlit.testing.v1 import AppTest

    processor = video.ASLProcessor(Mock(), threading.Lock(), "A")
    context = SimpleNamespace(video_processor=processor)
    monkeypatch.setattr(ui, "webrtc_streamer", lambda **kwargs: context)
    monkeypatch.setattr(ui, "get_ice_servers", lambda: [])
    page = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    assert not page.exception
    assert page.text_area[0].value == "A"
    page.text_area[0].input("EDITED").run()
    assert not page.exception
    assert processor.snapshot()[0] == "EDITED"
    processor.set_sentence("LIVE")
    page.run()
    assert page.text_area[0].value == "LIVE"
    next(button for button in page.button if "Clear" in button.label).click().run()
    assert not page.exception
    assert page.text_area[0].value == processor.snapshot()[0] == ""
