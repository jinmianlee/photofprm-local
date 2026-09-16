import json
import sys
import traceback
import os
import time
from pathlib import Path
from .settings import write_json
from .file_io import read_json
from .job_timing import timed_phase


def main(job):
    job = Path(job)
    request = json.loads((job / 'request.json').read_text('utf-8'))
    started = time.monotonic()
    started_at = time.time()
    last_progress = 0
    last_update = None
    def update(progress, message):
        nonlocal last_progress, last_update
        progress = max(last_progress, progress)
        if last_update == (progress, message):
            return
        try:
            write_json(job / 'status.json', {'id': job.name, 'kind': request['kind'],
                       'state': 'running', 'progress': progress, 'message': message,
                       'started_at': started_at, 'elapsed_seconds': round(time.monotonic()-started, 1)})
            last_progress, last_update = progress, (progress, message)
        except OSError as error:
            # A progress display failure must not discard minutes of valid
            # inference. Critical result/final-state writes still propagate.
            print(f'Progress update delayed; computation continues: {error}', flush=True)
    try:
        try:
            write_json(job / 'timings.json', {'phases': []})
        except OSError as error:
            print(f'Timing report unavailable: {error}', flush=True)
        from .geometry import make_demo, process_mesh
        update(3, '载入任务')
        source = job / 'source.ply'
        provenance = None
        if request['kind'] == 'demo':
            make_demo(source)
        elif request['kind'] == 'photos':
            from .reconstruct import reconstruct
            source = reconstruct(job, request['options'], update)
        elif request['kind'] == 'single':
            from .single_photo import reconstruct
            source, provenance = reconstruct(job, request['options'], update, resume=request.get('resume', False))
        else:
            source = job / request['source_file']
            if (job / 'generation.json').is_file():
                provenance = json.loads((job / 'generation.json').read_text('utf-8'))
            if (job / 'generated_raw.ply').is_file() and (job / 'foreground.png').is_file():
                from .photo_color import prepare_colored_bust
                update(50, '重新对齐照片颜色')
                provenance = prepare_colored_bust(job / 'generated_raw.ply', job / 'foreground.png', source,
                    request['options'].get('photo_pitch_deg', 25), request['options'].get('photo_yaw_deg', 7),
                    color_style=request['options'].get('color_style', 'photo'), colors=request['options'].get('colors', 4),
                    photo_auto_align=request['options'].get('photo_auto_align', True),
                    palette_override=request['options'].get('palette_override'),
                    repair_small_holes=request['options'].get('repair_small_holes', False),
                    size_mm=request['options'].get('size_mm', 95))
        if provenance is not None and (job / 'input_region.json').is_file():
            provenance['input_region'] = read_json(job / 'input_region.json')
            provenance.setdefault('warnings', []).append('本次仅生成框选部位的独立模型，未与整图模型自动拼接。')
            write_json(job / 'generation.json', provenance)
        with timed_phase(job, 'color_partition_and_export'):
            report = process_mesh(source, job / 'output', request['options'], update, provenance=provenance)
        if (job / 'timings.json').is_file():
            try:
                report['timings'] = read_json(job / 'timings.json')
            except (OSError, ValueError) as error:
                print(f'Timing report unavailable: {error}', flush=True)
        write_json(job / 'status.json', {'id': job.name, 'kind': request['kind'],
                   'state': 'complete', 'progress': 100, 'message': '模型已生成，请在切片软件中检查细节与耗材分配',
                   'report': report, 'started_at': started_at,
                   'elapsed_seconds': round(time.monotonic()-started, 1)})
    except Exception as error:
        traceback.print_exc()
        write_json(job / 'status.json', {'id': job.name, 'kind': request['kind'],
                   'state': 'failed', 'progress': 0, 'message': str(error)[:1500]})


if __name__ == '__main__':
    from .process_tree import wait_for_start
    wait_for_start()
    # Keep Windows task logs readable and constrain numeric thread pools.
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main(sys.argv[1])
