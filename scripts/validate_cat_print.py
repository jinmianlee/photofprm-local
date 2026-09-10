"""Run the real photo-generated mesh through local color/print preparation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.photo_color import prepare_colored_bust
from backend.geometry import process_mesh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mesh', required=True)
parser.add_argument('--foreground', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--color-only', action='store_true')
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
source = output / 'source.ply'
provenance = prepare_colored_bust(Path(args.mesh), Path(args.foreground), source)
print(json.dumps(provenance, ensure_ascii=False), flush=True)
if not args.color_only:
    report = process_mesh(source, output / 'output',
                          {'size_mm': 95, 'colors': 4, 'pitch_mm': .8,
                           'min_feature_mm': .8, 'repair_small_holes': False},
                          lambda *args: print(args, flush=True), provenance=provenance)
    print(json.dumps(report, ensure_ascii=False), flush=True)
