"""Probe Bambu's material metadata using a copied, non-user test mesh."""
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'data/tests-single-02/colored0/output/colored.3mf'
target = ROOT / 'data/slicer-validation/metadata-test.3mf'
ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
with zipfile.ZipFile(source) as archive:
    model = ET.fromstring(archive.read('3D/3dmodel.model'))
    assembly = model.find('m:build/m:item', ns).get('objectid')
    objects = model.findall('m:resources/m:object', ns)
    meshes = [obj for obj in objects if obj.find('m:mesh', ns) is not None]
    colors = [base.get('displaycolor')[:7] for base in model.findall('m:resources/m:basematerials/m:base', ns)]
    config = ET.Element('config')
    obj = ET.SubElement(config, 'object', id=assembly)
    ET.SubElement(obj, 'metadata', key='name', value='PhotoForm assembly')
    for index, part in enumerate(meshes):
        entry = ET.SubElement(obj, 'part', id=part.get('id'), subtype='normal_part')
        ET.SubElement(entry, 'metadata', key='name', value=f'Color {index+1}')
        ET.SubElement(entry, 'metadata', key='extruder', value=str(index+1))
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
        for item in archive.infolist():
            output.writestr(item, archive.read(item.filename))
        output.writestr('Metadata/model_settings.config', ET.tostring(config, encoding='utf-8', xml_declaration=True))
        output.writestr('Metadata/project_settings.config', json.dumps({'filament_colour': colors}))
print(target)
