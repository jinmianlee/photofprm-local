"""Own each worker tree, including Conda children, without administrative taskkill.

Windows Job Objects: https://learn.microsoft.com/windows/win32/procthread/job-objects
Workers attach themselves before loading models; a ready event closes the startup race.
"""
import os
import subprocess
import time
import uuid

if os.name == 'nt':
    import ctypes as C
    from ctypes import wintypes as W
    K = C.WinDLL('kernel32', use_last_error=True)
    for name, args, result in [
        ('CreateJobObjectW', [C.c_void_p, W.LPCWSTR], W.HANDLE),
        ('OpenJobObjectW', [W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE),
        ('SetInformationJobObject', [W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
        ('QueryInformationJobObject', [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
        ('AssignProcessToJobObject', [W.HANDLE, W.HANDLE], W.BOOL),
        ('IsProcessInJob', [W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
        ('TerminateJobObject', [W.HANDLE, W.UINT], W.BOOL),
        ('GetCurrentProcess', [], W.HANDLE),
        ('CreateEventW', [C.c_void_p, W.BOOL, W.BOOL, W.LPCWSTR], W.HANDLE),
        ('OpenEventW', [W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE),
        ('SetEvent', [W.HANDLE], W.BOOL),
        ('WaitForSingleObject', [W.HANDLE, W.DWORD], W.DWORD),
        ('CloseHandle', [W.HANDLE], W.BOOL),
    ]:
        function = getattr(K, name)
        function.argtypes, function.restype = args, result

    class BasicLimits(C.Structure):
        _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64), ('flags', W.DWORD),
                    ('min_ws', C.c_size_t), ('max_ws', C.c_size_t), ('active', W.DWORD),
                    ('affinity', C.c_size_t), ('priority', W.DWORD), ('scheduling', W.DWORD)]

    class ExtendedLimits(C.Structure):
        _fields_ = [('basic', BasicLimits), ('io', C.c_uint64 * 6),
                    ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
                    ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]

    class Accounting(C.Structure):
        _fields_ = [('times', C.c_int64 * 4), ('page_faults', W.DWORD), ('total', W.DWORD),
                    ('active', W.DWORD), ('terminated', W.DWORD)]

    def checked(value):
        if not value:
            raise C.WinError(C.get_last_error())
        return value


def wait_for_start():
    """Called before a worker can spawn children; direct CLI runs need no owner."""
    if os.name != 'nt':
        return
    name = os.environ.pop('PHOTOFORM_WORKER_JOB', None)
    event_name = os.environ.pop('PHOTOFORM_WORKER_READY', None)
    if not name:
        return
    job = checked(K.OpenJobObjectW(0x0001 | 0x0004, False, name))
    event = None
    try:
        in_job = W.BOOL()
        checked(K.IsProcessInJob(K.GetCurrentProcess(), job, C.byref(in_job)))
        if not in_job.value:
            checked(K.AssignProcessToJobObject(job, K.GetCurrentProcess()))
        event = checked(K.OpenEventW(0x0002, False, event_name))
        checked(K.SetEvent(event))
    finally:
        if event:
            K.CloseHandle(event)
        K.CloseHandle(job)


class ManagedProcess(subprocess.Popen):
    def __init__(self, args, **kwargs):
        self.job_handle = self.ready_handle = None
        if os.name != 'nt':
            kwargs['start_new_session'] = True
            super().__init__(args, **kwargs)
            return
        name = 'Local\\PhotoForm-'+uuid.uuid4().hex
        self.job_handle = checked(K.CreateJobObjectW(None, name))
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            checked(K.SetInformationJobObject(self.job_handle, 9, C.byref(limits), C.sizeof(limits)))
            self.ready_handle = checked(K.CreateEventW(None, True, False, name+'-ready'))
            environment = dict(kwargs.pop('env', os.environ))
            environment.update(PHOTOFORM_WORKER_JOB=name, PHOTOFORM_WORKER_READY=name+'-ready')
            super().__init__(args, env=environment, **kwargs)
            if K.WaitForSingleObject(self.ready_handle, 15000) != 0:
                raise RuntimeError('Worker failed to attach to its process group; see worker.log.')
        except BaseException:
            if self.job_handle:
                K.TerminateJobObject(self.job_handle, 1)
            if getattr(self, '_child_created', False):
                super().kill()
                super().wait(timeout=10)
            self.close_handles()
            raise

    def close_handles(self):
        if os.name == 'nt':
            for name in ('ready_handle', 'job_handle'):
                handle = getattr(self, name, None)
                if handle:
                    K.CloseHandle(handle)
                    setattr(self, name, None)

    def active_children(self):
        if not self.job_handle:
            return 0
        info = Accounting()
        checked(K.QueryInformationJobObject(self.job_handle, 1, C.byref(info), C.sizeof(info), None))
        return info.active

    def poll(self):
        result = super().poll()
        if result is not None and os.name == 'nt' and self.job_handle:
            if self.active_children():
                checked(K.TerminateJobObject(self.job_handle, 1))
                return None
            self.close_handles()
        return result

    def terminate_tree(self):
        if self.poll() is not None:
            return
        if os.name == 'nt':
            if self.job_handle:
                checked(K.TerminateJobObject(self.job_handle, 1))
            super().wait(timeout=10)
            deadline = time.monotonic()+10
            while self.active_children():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Worker children are still stopping; retry cancellation.')
                time.sleep(.05)
            self.close_handles()
        else:
            import signal
            os.killpg(self.pid, signal.SIGTERM)
            try:
                self.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.pid, signal.SIGKILL)
                self.wait(timeout=10)
