"""Kokoro downloads a missing voice even when HF offline mode is armed.

pipeline() returns a generator. The missing voices/*.pt error is raised
on the first iteration, so the retry has to pull that item itself.
"""
import os
import sys
import types

from core.tts.hf_offline import hf_go_online, prime_generator, voice_needs_download


class LocalEntryNotFoundError(Exception):
    pass


class OfflineModeIsEnabled(Exception):
    pass


def test_prime_generator_raises_on_the_first_item():
    def gen():
        raise LocalEntryNotFoundError("voices/af_sky.pt")
        yield 1  # pragma: no cover

    try:
        prime_generator(gen())
    except LocalEntryNotFoundError:
        return
    raise AssertionError("first-item failure must surface inside prime_generator")


def test_prime_generator_replays_the_first_item():
    def gen():
        yield ("a", "b", 1)
        yield ("c", "d", 2)

    assert list(prime_generator(gen())) == [("a", "b", 1), ("c", "d", 2)]


def test_missing_voice_retries_after_offline_is_lifted(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = {"n": 0}

    def pipeline():
        calls["n"] += 1
        if calls["n"] == 1:
            raise LocalEntryNotFoundError("not in cache") from OfflineModeIsEnabled("offline")
        yield ("g", "p", "audio")

    def synth():
        def _open():
            return prime_generator(pipeline())

        try:
            return _open()
        except Exception as e:
            if voice_needs_download(e):
                hf_go_online()
                return _open()
            raise

    hub = types.ModuleType("huggingface_hub.constants")
    hub.HF_HUB_OFFLINE = True
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hub)

    assert list(synth()) == [("g", "p", "audio")]
    assert calls["n"] == 2
    assert "HF_HUB_OFFLINE" not in os.environ
    assert hub.HF_HUB_OFFLINE is False


def test_real_synth_error_is_not_a_download():
    os.environ.pop("HF_HUB_OFFLINE", None)
    assert voice_needs_download(RuntimeError("boom")) is False
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        assert voice_needs_download(RuntimeError("boom")) is False
        assert voice_needs_download(LocalEntryNotFoundError("missing")) is True
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
