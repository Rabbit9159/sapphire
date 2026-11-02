"""Hugging Face offline helpers for the Kokoro process.

The model snapshot can already be cached while one voice file is not.
Kokoro's pipeline() returns a generator, and the missing-file error is
raised on the first iteration, not on the call. prime_generator pulls
that first item so the caller can lift offline mode and retry.
"""
import os
import sys

_FETCH_ERRORS = (
    'OfflineModeIsEnabled',
    'LocalEntryNotFoundError',
    'FileNotFoundError',
)


def hf_go_online():
    """Lift HF offline mode for the rest of this process.

    huggingface_hub reads HF_HUB_OFFLINE once into constants.HF_HUB_OFFLINE
    and is_offline_mode() returns that flag. Unset the env var and flip
    every copy already imported. A restart re-arms offline mode.
    """
    os.environ.pop('HF_HUB_OFFLINE', None)
    for name, mod in list(sys.modules.items()):
        if name.startswith('huggingface_hub') and hasattr(mod, 'HF_HUB_OFFLINE'):
            try:
                setattr(mod, 'HF_HUB_OFFLINE', False)
            except Exception:
                pass


def voice_needs_download(exc):
    """True when this failure is a voice file the offline cache doesn't have."""
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        return False
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__
        text = str(cur).lower()
        if (name in _FETCH_ERRORS or 'offline' in text
                or 'local_files_only' in text):
            return True
        cur = cur.__cause__
    return False


def prime_generator(gen):
    """Pull the first item now. Replay it, then the rest, to the caller."""
    try:
        first = next(gen)
    except StopIteration:
        return iter(())

    def _rest():
        yield first
        yield from gen

    return _rest()
