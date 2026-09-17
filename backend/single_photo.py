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
from .runtime_config import config, fingerprint

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
    try:
        c = config()
        current = fingerprint(c)
        weight = c['model_dir'] / 'model.fp16.safetensors'
        installed = all(f['size'] is not None for f in current['files']) and weight.stat().st_size == 3822584202
        check = read_json(DATA / 'engine-check.json')
        ready = bool(installed and check.get('ready') is True and check.get('fingerprint') == current)
        message = '本地单图引擎就绪' if ready else ('文件已找到，请运行“检查模型环境.cmd”验证当前环境。' if installed else '单图环境或模型文件缺失，请按 DEPLOYMENT.md 安装或配置已有路径。')
        if not ready and check.get('error'):
            message += ' 上次检查：' + str(check['error'])[:300]
        size = weight.stat().st_size if weight.is_file() else 0
    except (OSError, ValueError, TypeError) as error:
        ready, installed, size = False, False, 0
        message = '本机 photoform.json 配置无法读取：' + str(error)[:250]
    shape21 = shape21_capability()
    return {'ready': ready, 'installed': bool(installed), 'model': 'Hunyuan3D-2mini-Turbo',
            'device': 'Intel Arc / XPU', 'download_bytes': size,
            'download_total': 3822584202,
            'download_state': 'verified' if ready else 'check_required', 'message': message,
            'shape21': shape21}


def shape21_capability():
    try:
        c = config(); weight = c['shape21_model_dir']/'model.fp16.ckpt'
        cfg = c['shape21_model_dir']/'config.yaml'
        source = c['shape21_source_dir']/'hy3dshape/hy3dshape/pipelines.py'
        installed = weight.is_file() and weight.stat().st_size == 7366389768 and cfg.is_file() and source.is_file()
    except (OSError, ValueError, TypeError):
        installed = False
    return {'ready': installed, 'installed': installed, 'model': 'Hunyuan3D-Shape-2.1',
            'message': '2.1 精细形状已就绪' if installed else '2.1 精细形状未安装；可继续使用 Mini。'}


def run_stage(arguments, update, job=None):
    c = config()
    command = [c['conda'], 'run', '--prefix', str(c['ai_env']), '--no-capture-output', 'python', '-u', *map(str, arguments)]
    environment = os.environ.copy()
    environment.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', U2NET_HOME=str(c['rembg_dir']))
    with subprocess.Popen(command, cwd=ROOT, env=environment,
                          stdout=sys.stdout, stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as process:
        while process.poll() is None:
            if job:
                status = read_json(job / 'inference.json')
                if status:
                    stage = status.get('stage', '生成三维形状')
                    if stage in ('推断三维形状', 'Infer official 2.1 shape'):
                        current, total = status.get('diffusion_step', 0), status.get('diffusion_steps', 30)
                        update(12 + round(current/max(total, 1)*25), f'推断三维形状 · {current}/{total} 步')
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
    engine = options.get('shape_engine', 'mini-turbo')
    available = capability()
    if engine == 'shape-2.1':
        if not available['shape21']['ready']:
            raise RuntimeError('本地 Hunyuan3D 2.1 精细形状模型尚未安装。')
    elif not available['ready']:
        raise RuntimeError('本地单图引擎尚未通过实际生成验证。')
    foreground = job / 'foreground.png'
    resolution = options.get('mesh_resolution', 255)
    steps, seed = options.get('shape_steps', 5), options.get('shape_seed', 12345)
    if not resume or not foreground.is_file():
        update(5, '本地去除照片背景')
        with timed_phase(job, 'foreground'):
            run_stage([ROOT / 'backend/foreground.py', '--input', job / 'images/photo_0000.png', '--output', foreground], update)
    with timed_phase(job, 'completed_shape_check'):
        reuse = resume and engine == 'mini-turbo' and completed_shape(job, foreground, resolution, steps, seed)
    if reuse:
        update(56, '复用已完成的原始网格，直接进行打印处理')
    else:
        update(10, '载入混元 2.1 精细形状' if engine == 'shape-2.1' else '载入混元 2mini')
        with timed_phase(job, 'shape_inference_and_extraction'):
            if engine == 'shape-2.1':
                c = config()
                run_stage([ROOT/'scripts/compare_shape21.py', '--input', foreground, '--output', job,
                           '--source', c['shape21_source_dir'], '--model', c['shape21_model_dir'],
                           '--resolution', str(resolution), '--steps', str(steps), '--seed', str(seed),
                           *(['--resume'] if resume else [])], update, job)
            else:
                run_stage([ROOT / 'backend/single_photo_turbo.py', '--input', foreground, '--output', job,
                           '--resolution', str(resolution), '--steps', str(steps), '--seed', str(seed), '--efficient-loading', *(['--resume'] if resume and (job/'diffusion_checkpoint.safetensors').is_file() else [])], update, job)
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
                                        size_mm=options.get('size_mm', 95), photo_landmarks=options.get('photo_landmarks'),
                                        model_name='Hunyuan3D-Shape-2.1' if engine == 'shape-2.1' else 'Hunyuan3D-2mini-Turbo')
    return source, provenance
