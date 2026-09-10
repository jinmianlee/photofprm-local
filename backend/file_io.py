"""Atomic local writes that tolerate brief Windows file sharing conflicts."""
import errno
import json
import os
from pathlib import Path
import tempfile
import time


def _sharing_retry(operation, timeout=2.0):
    deadline = time.monotonic() + timeout
    delay = .02
    while True:
        try:
            return operation()
        except OSError as error:
            transient = (getattr(error, 'winerror', None) in (5, 32, 33)
                         or isinstance(error, PermissionError) and error.errno in (errno.EACCES, errno.EPERM))
            if not transient or time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, max(0, deadline-time.monotonic())))
            delay = min(delay*2, .2)


def replace_with_retry(source, target, timeout=2.0):
    return _sharing_retry(lambda: os.replace(source, target), timeout)


def read_json(path):
    # On Windows an open can also encounter a file that is pending rename.
    return _sharing_retry(lambda: json.loads(Path(path).read_text('utf-8')))


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    # Different writers must never share a fixed status.tmp filename.
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(payload)
        replace_with_retry(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            # A failed final replacement preserves the existing complete file.
            pass
