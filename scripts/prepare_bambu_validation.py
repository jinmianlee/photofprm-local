"""Expand installed official presets for offline CLI checks, never printing."""
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
root = Path('C:/Bambu Studio/resources/profiles/BBL')
destination = Path(args.output)
destination.mkdir(parents=True, exist_ok=True)
index = {}
for path in root.rglob('*.json'):
    value = json.loads(path.read_text('utf-8'))
    if value.get('name'):
        index[value['name']] = value

def expand(name, seen=()):
    if name in seen:
        raise ValueError('Circular preset inheritance')
    data = index[name]
    result = expand(data['inherits'], seen+(name,)) if data.get('inherits') else {}
    for include in data.get('include', []):
        result.update(expand(include, seen+(name,)))
    result.update({k: v for k, v in data.items() if k not in ('inherits', 'include')})
    return result

def write(name, value):
    (destination / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), 'utf-8')

machine = expand('Bambu Lab A1 0.4 nozzle')
process = expand('0.16mm High Quality @BBL A1')
# Keep the wipe tower inside the test plate and clear of the centered bust.
process.update(wipe_tower_x=['15'], wipe_tower_y=['25'])
filament = expand('Generic PLA @BBL A1')
write('machine.json', machine)
write('process.json', process)
parts = json.loads(Path(args.report).read_text('utf-8'))['parts']
for i, part in enumerate(parts, 1):
    write(f'filament-{i}.json', {**filament, 'filament_colour': [part['color']]})
write('VALIDATION_ONLY.json', {'purpose': 'Offline slicing, not a confirmed user printer profile',
      'machine': machine['name'], 'layer_height': process['layer_height'],
      'printable_area': machine['printable_area'], 'filament_density': filament['filament_density'],
      'colors': [p['color'] for p in parts]})
print((destination / 'VALIDATION_ONLY.json').read_text('utf-8'))
