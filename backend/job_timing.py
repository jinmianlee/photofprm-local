"""Per-stage wall-clock timings for local jobs, separate from mesh validity."""
from contextlib import contextmanager
import json
import time
from .file_io import write_json


@contextmanager
def timed_phase(job, name):
    started = time.monotonic()
    state = 'complete'
    try:
        yield
    except BaseException:
        state = 'failed'
        raise
    finally:
        try:
            target = job / 'timings.json'
            data = json.loads(target.read_text('utf-8')) if target.is_file() else {'phases': []}
            data['phases'].append({'phase': name, 'seconds': round(time.monotonic()-started, 2), 'state': state})
            write_json(target, data)
        except (OSError, ValueError) as error:
            print(f'Timing report unavailable: {error}', flush=True)
