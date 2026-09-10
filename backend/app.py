from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from urllib.parse import urlparse
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from PIL import Image, ImageOps, ImageFilter, ImageStat
from .settings import ROOT, DATA, engine_path, write_json
from .file_io import read_json
from .photo_region import PhotoRegion

PROCESS_LOCK = threading.Lock()
PROCESSES = {}
ALLOWED_HOSTS = {'127.0.0.1', 'localhost', '[::1]'}
MAX_UPLOAD = 1024 * 1024 * 1024
MAX_FILE = 200 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    size_mm: float = Field(100, ge=10, le=500)
    colors: int = Field(4, ge=1, le=8)
    pitch_mm: float = Field(0.8, ge=0.15, le=3)
    min_feature_mm: float = Field(0.8, ge=0.2, le=5)
    image_size: int = Field(2400, ge=1200, le=6000)
    repair_small_holes: bool = False
    photo_pitch_deg: float = Field(25, ge=-45, le=60)
    photo_yaw_deg: float = Field(7, ge=-60, le=60)
    color_style: Literal['photo', 'flat'] = 'flat'
    photo_auto_align: bool = True
    mesh_resolution: Literal[255, 383] = 255
    palette_override: list[Annotated[str, Field(pattern=r'^#[0-9a-fA-F]{6}$')]] = Field(default_factory=list, max_length=8)


def parse_options(raw):
    try:
        return Options.model_validate_json(raw).model_dump()
    except ValidationError as e:
        raise HTTPException(422, '参数格式或取值范围不正确') from e


def job_dir(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404, '任务不存在')
    path = DATA / job_id
    if not path.is_dir():
        raise HTTPException(404, '任务不存在')
    return path


def state(path):
    result = read_json(path / 'status.json')
    with PROCESS_LOCK:
        process = PROCESSES.get(path.name)
        if result['state'] in ('running', 'queued') and (process is None or process.poll() is not None):
            # The worker can finish between the first read and poll(). Never
            # overwrite its newly published completion with a stale failure.
            result = read_json(path / 'status.json')
            if result['state'] in ('running', 'queued'):
                result.update(state='failed', message='任务进程已停止，可能因内存不足或应用重启。照片与日志仍保存在本地。', progress=0)
                write_json(path / 'status.json', result)
    if (path / 'capture_report.json').exists():
        result['capture'] = json.loads((path / 'capture_report.json').read_text('utf-8'))
    result['source_available'] = any((path / ('source'+ext)).exists() for ext in ('.ply', '.glb', '.stl'))
    result['source_file'] = next((f'source{ext}' for ext in ('.ply', '.glb', '.stl') if (path / ('source'+ext)).exists()), None)
    result['can_resume'] = (result.get('kind') == 'single' and result['state'] in ('failed', 'cancelled')
                            and (path / 'diffusion_checkpoint.safetensors').is_file())
    result['raw_shape_available'] = (path / 'generated_raw.ply').is_file()
    result['can_refine'] = (result['state'] == 'complete' and (path / 'foreground.png').is_file()
                            and (path / 'diffusion_checkpoint.safetensors').is_file())
    return result


def launch(path, request):
    with PROCESS_LOCK:
        if any(p.poll() is None for p in PROCESSES.values()):
            raise HTTPException(409, '已有模型任务运行中，请等待完成或先取消该任务。')
        write_json(path / 'request.json', request)
        write_json(path / 'status.json', {'id': path.name, 'kind': request['kind'], 'state': 'queued', 'progress': 0, 'message': '任务已提交'})
        with (path / 'worker.log').open('wb') as log:
            PROCESSES[path.name] = subprocess.Popen([sys.executable, '-u', '-m', 'backend.worker', str(path)],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    return {'id': path.name}


def stop_process(process):
    if process.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@asynccontextmanager
async def lifespan(app):
    yield
    for p in PROCESSES.values():
        stop_process(p)


app = FastAPI(title='PhotoForm Local', lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, error):
    # Keep NaN/Infinity and raw uploaded values out of JSON error responses.
    return JSONResponse({'detail': '参数格式或取值范围不正确'}, status_code=422)


@app.middleware('http')
async def local_only(request: Request, call_next):
    # Protect local file-processing endpoints against cross-site form submission
    # and DNS rebinding. No wildcard CORS. All browser API calls are same-origin.
    hostname = request.url.hostname
    if hostname not in ALLOWED_HOSTS:
        return JSONResponse({'detail': '仅允许从本机 localhost 访问'}, status_code=403)
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        if request.headers.get('x-photoform-client') != 'local':
            return JSONResponse({'detail': '缺少本地客户端标识'}, status_code=403)
        if origin and (urlparse(origin).hostname not in ALLOWED_HOSTS or
                       request.headers.get('sec-fetch-site') == 'cross-site'):
            return JSONResponse({'detail': '拒绝跨站点请求'}, status_code=403)
    length = request.headers.get('content-length')
    if length:
        try:
            if int(length) > MAX_UPLOAD + 2_000_000:
                return JSONResponse({'detail': '单次上传上限为 1 GB'}, status_code=413)
        except ValueError:
            return JSONResponse({'detail': '无效请求长度'}, status_code=400)
    return await call_next(request)


@app.get('/api/system')
def system():
    from .single_photo import capability
    names = ['InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh']
    engines = {name: bool(engine_path(name)) for name in names}
    return {'app': 'PhotoForm Local', 'version': '0.1.0', 'local': True,
            'mesh_ready': all(importlib.util.find_spec(x) for x in ('trimesh','manifold3d','scipy','skimage')),
            'reconstruction_ready': bool(importlib.util.find_spec('pycolmap')) and all(engines.values()),
            'engines': engines, 'backend': 'COLMAP + OpenMVS · CPU',
            'gaussian_training': False, 'single_photo': capability()}


@app.get('/api/jobs')
def jobs():
    paths = sorted(DATA.glob('*/status.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    return [{k: v for k, v in state(p.parent).items() if k != 'report'} for p in paths]


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return state(job_dir(job_id))


@app.post('/api/demo')
def demo(options: Options):
    path = DATA / uuid.uuid4().hex
    path.mkdir()
    return launch(path, {'kind': 'demo', 'options': options.model_dump()})


@app.post('/api/upload')
async def upload(files: Annotated[list[UploadFile], File()],
                 options: Annotated[str, Form()], kind: Annotated[str, Form()],
                 region: Annotated[str | None, Form()] = None):
    opts = parse_options(options)
    selected_region = None
    if region is not None:
        if kind != 'single':
            raise HTTPException(422, '框选部位仅用于单张照片的独立建模。')
        try:
            selected_region = PhotoRegion.model_validate_json(region)
        except ValidationError as error:
            raise HTTPException(422, '框选区域无效，请重新选择。') from error
    if kind not in ('single', 'photos', 'mesh'):
        raise HTTPException(422, '未知输入类型')
    if kind == 'single' and len(files) != 1:
        raise HTTPException(422, '单图生成请上传一张清晰的主体照片。')
    if kind == 'photos' and not 8 <= len(files) <= 300:
        raise HTTPException(422, '请上传 8–300 张同一物体的多角度照片；建议 60–150 张。')
    if kind == 'mesh' and len(files) != 1:
        raise HTTPException(422, '一次导入一个 PLY、GLB 或 STL 网格。')
    if kind == 'photos' and not system()['reconstruction_ready']:
        raise HTTPException(409, '本机尚未安装 OpenMVS，请先运行安装重建引擎脚本。')
    if kind == 'single' and not system()['single_photo']['ready']:
        raise HTTPException(409, '混元 2mini 尚未通过本机生成验证，请等待安装和实测完成。')
    if any(p.poll() is None for p in PROCESSES.values()):
        raise HTTPException(409, '已有任务正在运行。')
    path = DATA / uuid.uuid4().hex
    path.mkdir()
    (path / 'images').mkdir()
    total = 0
    image_quality = []
    source_name = ''
    try:
        for i, file in enumerate(files):
            suffix = Path(file.filename or '').suffix.lower()
            if kind == 'mesh' and suffix not in ('.ply', '.glb', '.stl'):
                raise HTTPException(422, '支持带顶点颜色的 PLY、内嵌纹理 GLB，或单色 STL。')
            if kind in ('single', 'photos') and suffix not in ('.jpg', '.jpeg', '.png', '.webp'):
                raise HTTPException(422, '照片支持 JPG、PNG、WebP；HEIC 请先转成 JPG。')
            # Names are generated by the server; original names never become paths.
            raw_path = path / (f'upload_{i:04d}' + suffix)
            count = 0
            with raw_path.open('wb') as output:
                while chunk := await file.read(1024*1024):
                    total += len(chunk)
                    count += len(chunk)
                    if total > MAX_UPLOAD or count > MAX_FILE:
                        raise HTTPException(413, '上传过大：单文件不超过 200 MB，总量不超过 1 GB。')
                    output.write(chunk)
            if kind in ('single', 'photos'):
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(raw_path) as img:
                        img = ImageOps.exif_transpose(img).convert('RGBA' if kind == 'single' else 'RGB')
                        minimum = 256 if kind == 'single' else 640
                        if min(img.size) < minimum:
                            raise HTTPException(422, f'第 {i+1} 张照片分辨率不足，短边至少 {minimum} 像素。')
                        if selected_region:
                            try:
                                img, region_record = selected_region.crop(img)
                            except ValueError as error:
                                raise HTTPException(422, str(error)) from error
                            write_json(path / 'input_region.json', region_record)
                        thumb = img.copy()
                        thumb.thumbnail((640,640))
                        sharpness = ImageStat.Stat(thumb.convert('L').filter(ImageFilter.FIND_EDGES)).var[0]
                        image_quality.append({'index': i+1, 'width': img.width, 'height': img.height,
                                              'edge_variance': round(sharpness, 2)})
                        if kind == 'single':
                            img.save(path / 'images' / f'photo_{i:04d}.png')
                        else:
                            img.save(path / 'images' / f'photo_{i:04d}.jpg', quality=98, subsampling=0)
            else:
                source_name = 'source' + suffix
                raw_path.rename(path / source_name)
        write_json(path / 'image_quality.json', image_quality)
        return launch(path, {'kind': kind, 'source_file': source_name, 'options': opts})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, '文件无法解析，请检查照片或模型格式。') from e
    finally:
        for file in files:
            await file.close()


@app.post('/api/jobs/{job_id}/reprocess')
def reprocess(job_id: str, options: Options):
    old = job_dir(job_id)
    request = json.loads((old / 'request.json').read_text('utf-8'))
    source = old / (request.get('source_file') or 'source.ply')
    if not source.is_file():
        raise HTTPException(409, '该任务尚未生成可处理的网格')
    path = DATA / uuid.uuid4().hex
    path.mkdir()
    shutil.copy2(source, path / source.name)
    if (old / 'generation.json').is_file():
        shutil.copy2(old / 'generation.json', path / 'generation.json')
    # Preserve photo-generation inputs so color alignment can be adjusted
    # without spending another inference run or mutating the original job.
    for name in ('generated_raw.ply', 'foreground.png', 'diffusion_checkpoint.safetensors', 'shape_result.json', 'inference.json', 'input_region.json'):
        if (old / name).is_file():
            shutil.copy2(old / name, path / name)
    return launch(path, {'kind': 'mesh', 'source_file': source.name, 'options': options.model_dump()})


@app.post('/api/jobs/{job_id}/refine')
def refine(job_id: str, options: Options):
    old = job_dir(job_id)
    if not state(old)['can_refine']:
        raise HTTPException(409, '需要已完成的单图模型及其形状断点。')
    path = DATA / uuid.uuid4().hex
    path.mkdir()
    for name in ('foreground.png', 'diffusion_checkpoint.safetensors'):
        shutil.copy2(old / name, path / name)
    if (old / 'input_region.json').is_file():
        shutil.copy2(old / 'input_region.json', path / 'input_region.json')
    return launch(path, {'kind': 'single', 'resume': True, 'source_job': old.name,
                        'options': options.model_dump()})


@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str):
    path = job_dir(job_id)
    process = PROCESSES.get(job_id)
    if process is None or process.poll() is not None:
        return state(path)
    stop_process(process)
    previous = state(path)
    previous.update(state='cancelled', progress=0, message='任务已取消，输入文件已保留。')
    write_json(path / 'status.json', previous)
    return previous


@app.post('/api/jobs/{job_id}/resume')
def resume(job_id: str, options: Options):
    path = job_dir(job_id)
    if not state(path)['can_resume']:
        raise HTTPException(409, '此任务没有可恢复的单图推理断点。')
    request = json.loads((path / 'request.json').read_text('utf-8'))
    if request.get('kind') != 'single':
        raise HTTPException(409, '只支持恢复单图生成任务。')
    request.update(resume=True, options=options.model_dump())
    return launch(path, request)


@app.get('/api/jobs/{job_id}/log')
def log(job_id: str):
    path = job_dir(job_id) / 'worker.log'
    if not path.exists():
        return {'text': ''}
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-12000))
        return {'text': f.read().decode('utf-8', errors='replace')}


@app.get('/api/jobs/{job_id}/files/{filename}')
def download(job_id: str, filename: str):
    path = job_dir(job_id)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', filename) or '..' in filename:
        raise HTTPException(404)
    if filename in ('source.ply', 'source.glb', 'source.stl'):
        target = path / filename
    else:
        if state(path)['state'] != 'complete':
            raise HTTPException(409, '模型尚未通过导出检查')
        target = path / 'output' / filename
    if not target.is_file():
        raise HTTPException(404, '文件不存在')
    return FileResponse(target, filename=filename)


if (ROOT / 'dist' / 'assets').exists():
    app.mount('/assets', StaticFiles(directory=ROOT / 'dist' / 'assets'), name='assets')


@app.get('/')
def index():
    target = ROOT / 'dist' / 'index.html'
    if not target.exists():
        raise HTTPException(503, '请先运行 npm run build 构建本地界面')
    return FileResponse(target)
