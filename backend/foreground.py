"""Local U2Net foreground extraction; downloading is an explicit install step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'rembg' / 'u2net.onnx'
URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'
MD5 = '60024c5c889badc19c04ad937298a77b'


def verify(path):
    if not path.is_file():
        raise RuntimeError('本地抠图模型尚未安装，请先运行 foreground.py --install。')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    if digest != MD5:
        raise RuntimeError('抠图权重与 rembg 官方校验值不一致，停止加载。')


def install():
    if MODEL.is_file():
        verify(MODEL)
        print('Foreground model already verified.', flush=True)
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL.with_suffix('.onnx.download')
    print('Downloading foreground model from the official rembg release.', flush=True)
    with urllib.request.urlopen(URL, timeout=30) as source, partial.open('wb') as target:
        total = 0
        while block := source.read(1024 * 1024):
            target.write(block)
            total += len(block)
            if total % (16 * 1024 * 1024) == 0:
                print(f'{total // 1048576} MiB downloaded', flush=True)
    verify(partial)
    partial.replace(MODEL)
    with MODEL.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    MODEL.with_suffix('.json').write_text(json.dumps({
        'url': URL, 'official_md5': MD5, 'sha256': sha,
        'bytes': MODEL.stat().st_size}, indent=2), 'utf-8')
    print('Foreground model verified.', flush=True)


def extract(source, target):
    # Reject missing/corrupt files before rembg can attempt an automatic download.
    verify(MODEL)
    os.environ['U2NET_HOME'] = str(MODEL.parent)
    os.environ['OMP_NUM_THREADS'] = '4'
    from PIL import Image, ImageOps
    import numpy as np
    from rembg import new_session, remove
    image = ImageOps.exif_transpose(Image.open(source)).convert('RGBA')
    if min(image.size) < 256:
        raise ValueError('照片短边至少需要 256 像素。')
    if image.getchannel('A').getextrema()[0] < 255:
        result = image
    else:
        session = new_session('u2net', providers=['CPUExecutionProvider'])
        result = remove(image, session=session, post_process_mask=True)
    alpha = np.asarray(result.getchannel('A'))
    coverage = float(np.mean(alpha >= 128))
    if not 0.03 < coverage < 0.98:
        raise ValueError('自动抠图未分离出清晰主体，请上传透明背景 PNG。')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target)
    # Keep this small report beside the original, uncropped mask for inspection.
    target.with_suffix('.json').write_text(json.dumps({
        'model': 'U2Net', 'local': True, 'size': list(result.size),
        'foreground_fraction': round(coverage, 4)}, indent=2), 'utf-8')
    print(f'Foreground saved: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.install:
        install()
    if args.input and args.output:
        extract(args.input, args.output)
    elif not args.install:
        parser.error('--input and --output are required for extraction')


if __name__ == '__main__':
    main()
