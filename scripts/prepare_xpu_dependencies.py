"""Install pinned inference dependencies in the optional Intel XPU Conda env.

PyTorch 2.6.0+XPU and torchvision 0.21.0+XPU must already be installed from
https://download.pytorch.org/whl/xpu. This script never changes the GPU driver.
"""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
prefix = root.parent / 'photo-to-print-xpu-conda'
conda = 'C:/miniconda3/Scripts/conda.exe'
command = [conda, 'run', '--prefix', str(prefix), '--no-capture-output', 'python']
pins = []
for line in (root / 'requirements-ai-installed.txt').read_text('utf-8').splitlines():
    name = line.partition('==')[0].lower().replace('_', '-')
    if name not in {'torch', 'torchvision', 'sympy'} and line.strip():
        pins.append(line)
target = root / 'data/xpu-dependencies.txt'
target.write_text('\n'.join(pins) + '\n', 'utf-8')
constraints = root / 'data/xpu-constraints.txt'
constraints.write_text('torch==2.6.0+xpu\ntorchvision==0.21.0+xpu\nsympy==1.13.1\n', 'utf-8')
environment = os.environ.copy()
environment.update(PYTHONUTF8='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
subprocess.run([*command, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
                '--timeout', '45', '--retries', '3', '-c', str(constraints), '-r', str(target)],
               cwd=root, env=environment, check=True)
subprocess.run([*command, '-m', 'pip', 'check'], cwd=root, env=environment, check=True)
frozen = subprocess.check_output([*command, '-m', 'pip', 'freeze'], cwd=root, env=environment, text=True)
(root / 'requirements-xpu-installed.txt').write_text(frozen, 'utf-8')
