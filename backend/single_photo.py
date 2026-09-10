"""Run the isolated Miniconda image-to-3D process and report real progress."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .settings import ROOT, DATA
from .shape_cache import completed_shape
from .job_timing import timed_phase
from .file_io import read_json as read_json_file

ENV = ROOT.parent / 'photo-to-print-xpu-conda'
CONDA = shutil.which('conda') or 'C:/miniconda3/Scripts/conda.exe'
WEIGHT = ROOT / 'models/hunyuan3d-2mini/hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors'
WEIGHT_SHA = 'bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4'


def read_json(path):
    try:
        return read_json_file(path)
    except (OSError, ValueError):
        return {}


def capability():
    validation = read_json(DATA / 'single-photo-validation.json')
    download = read_json(DATA / 'weight-download.json')
    installed = (ENV / 'python.exe').is_file() and WEIGHT.is_file() and WEIGHT.stat().st_size == 3822584202
    installed = installed and (ROOT / 'models/rembg/u2net.onnx').is_file()
    ready = bool(installed and validation.get('inference_validated') is True
                 and validation.get('print_pipeline_validated') is True
                 and validation.get('weight_sha256') == WEIGHT_SHA)
    return {'ready': ready, 'installed': bool(installed), 'model': 'Hunyuan3D-2mini-Turbo',
            'device': 'Intel Arc / XPU', 'download_bytes': WEIGHT.stat().st_size if WEIGHT.is_file() else 0,
            'download_total': 3822584202,
            'download_state': 'verified' if installed else download.get('state', 'pending'),
            'message': '本地单图引擎就绪' if ready else '文件已安装，正在验证猫照推理' if installed else '正在安装本地单图引擎'}


def run_stage(arguments, update, job=None):
    command = [CONDA, 'run', '--prefix', str(ENV), '--no-capture-output', 'python', '-u', *map(str, arguments)]
    environment = os.environ.copy()
    environment.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', U2NET_HOME=str(ROOT / 'models/rembg'))
    with subprocess.Popen(command, cwd=ROOT, env=environment,
                          stdout=sys.stdout, stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as process:
        while process.poll() is None:
            if job:
                status = read_json(job / 'inference.json')
                if status:
                    stage = status.get('stage', '生成三维形状')
                    if stage == '推断三维形状':
                        current, total = status.get('diffusion_step', 0), status.get('diffusion_steps', 30)
                        update(12 + round(current/max(total, 1)*25), f'{stage} · {current}/{total} 步')
                    elif stage == '细化三维表面':
                        resolution = status.get('grid_resolution', 63)
                        fraction = status.get('samples_done', 0)/max(status.get('samples_total', 1), 1)
                        progress = 38 + (0 if resolution <= 63 else 6 if resolution <= 127 else 12) + fraction*5
                        update(round(progress), f'{stage} · {resolution} 格 · {fraction:.0%}')
                    elif status.get('state') == 'shape_generated':
                        update(56, stage)
                    elif stage == '生成三角网格':
                        update(38, stage)
                    else:
                        update(10, stage)
            time.sleep(2)
        if process.returncode:
            status = read_json(job / 'inference.json') if job else {}
            raise RuntimeError(status.get('error') or '单张照片处理失败，请查看本地任务日志。')


def reconstruct(job, options, update, resume=False):
    if not capability()['ready']:
        raise RuntimeError('本地单图引擎尚未通过实际生成验证。')
    foreground = job / 'foreground.png'
    resolution = options.get('mesh_resolution', 255)
    if not resume or not foreground.is_file():
        update(5, '本地去除照片背景')
        with timed_phase(job, 'foreground'):
            run_stage([ROOT / 'backend/foreground.py', '--input', job / 'images/photo_0000.png', '--output', foreground], update)
    with timed_phase(job, 'completed_shape_check'):
        reuse = resume and completed_shape(job, foreground, resolution)
    if reuse:
        update(56, '复用已完成的原始网格，直接进行打印处理')
    else:
        update(10, '载入混元 2mini')
        with timed_phase(job, 'shape_inference_and_extraction'):
            run_stage([ROOT / 'backend/single_photo_turbo.py', '--input', foreground, '--output', job,
                       '--resolution', str(resolution), '--efficient-loading', *(['--resume'] if resume else [])], update, job)
    update(57, '整理底面并投影照片颜色')
    from .photo_color import prepare_colored_bust
    source = job / 'source.ply'
    with timed_phase(job, 'color_projection_and_base'):
        provenance = prepare_colored_bust(job / 'generated_raw.ply', foreground, source,
                                        options.get('photo_pitch_deg', 25), options.get('photo_yaw_deg', 7),
                                        color_style=options.get('color_style', 'photo'), colors=options.get('colors', 4),
                                        photo_auto_align=options.get('photo_auto_align', True),
                                        palette_override=options.get('palette_override'),
                                        repair_small_holes=options.get('repair_small_holes', False),
                                        size_mm=options.get('size_mm', 95))
    return source, provenance
