import os
import sys
import contextlib
import warnings
import logging

def suppress_pydub_warnings():
    """Filters out specific pydub syntax warnings."""
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

def configure_logging():
    """Silences library loggers."""
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    logging.getLogger("realtime_stt").setLevel(logging.ERROR)
    # Filter ctranslate2 if accessible via python logging, though often it prints to C-stderr

@contextlib.contextmanager
def ignore_stderr():
    """
    Context manager that suppresses C-level stderr output (ALSA, CTranslate2, etc).
    It redirects file descriptor 2 (stderr) to /dev/null temporarily.
    """
    # Don't suppress if we are debugging or on Windows (os.devnull works differently there sometimes)
    if os.environ.get("DEBUG"):
        yield
        return

    try:
        # Open null device
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Save original stderr
        old_stderr = os.dup(2)
        sys.stderr.flush()
        
        # Redirect stderr to null
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            # Restore stderr
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)
    except Exception:
        # If OS operations fail (e.g. some restricted envs), just yield
        yield