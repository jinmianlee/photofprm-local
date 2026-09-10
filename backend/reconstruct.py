from pathlib import Path
import os
import subprocess
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from .settings import engine_path, write_json


REQUIRED_ENGINES = ['InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh']


def reconstruct(job, options, update):
    import pycolmap
    missing = [x for x in REQUIRED_ENGINES if not engine_path(x)]
    if missing:
        raise ValueError('尚未安装 OpenMVS 重建引擎：' + ', '.join(missing) + '。请运行安装重建引擎脚本，或导入已有彩色网格。')
    job = Path(job)
    images = job / 'images'
    sparse = job / 'sparse'
    sparse.mkdir(exist_ok=True)
    database = job / 'features.db'
    update(8, '提取照片特征（CPU），保留原始上传照片')
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = options['image_size']
    extraction.num_threads = min(os.cpu_count() or 4, 8)
    pycolmap.extract_features(database_path=str(database), image_path=str(images),
                            camera_mode=pycolmap.CameraMode.AUTO,
                            extraction_options=extraction, device=pycolmap.Device.cpu)
    update(17, '匹配照片中的共同特征')
    pycolmap.match_exhaustive(database_path=str(database), device=pycolmap.Device.cpu)
    update(25, '估计相机位置与稀疏结构')
    reconstructions = pycolmap.incremental_mapping(database_path=str(database),
                            image_path=str(images), output_path=str(sparse))
    if not reconstructions:
        raise ValueError('没有重建出相机。请使用同一静止物体的多角度清晰照片，增加相邻视角重叠并避免纯色、反光表面。')
    best_id = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    best = reconstructions[best_id]
    total = len(list(images.glob('*.jpg')))
    registered = best.num_reg_images()
    write_json(job / 'capture_report.json', {'uploaded': total, 'registered': registered,
               'registered_fraction': registered/total, 'points3D': best.num_points3D()})
    if registered < 8 or registered < total*0.6:
        raise ValueError(f'仅有 {registered}/{total} 张照片成功定位，视角覆盖不足。已保存稀疏重建，请补拍后重试。')
    update(33, f'{registered}/{total} 张照片定位成功，校正镜头畸变')
    dense = job / 'dense'
    pycolmap.undistort_images(output_path=str(dense), input_path=str(sparse / str(best_id)),
            image_path=str(images), output_type='COLMAP',
            undistort_options={'max_image_size': options['image_size']})

    def run(name, args, percent, message):
        update(percent, message)
        command = [engine_path(name), *args]
        print('Running:', command, flush=True)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        result = subprocess.run(command, cwd=str(job), shell=False, creationflags=creationflags)
        # Windows release writes native logs to files rather than redirected stdout.
        native_logs = sorted(job.glob(name+'-*.log'), key=lambda p: p.stat().st_mtime)
        if native_logs:
            print(native_logs[-1].read_text('utf-8', errors='replace')[-16000:], flush=True)
        if result.returncode:
            raise ValueError(f'{name} 运行失败（退出码 {result.returncode}），请查看任务日志。')

    run('InterfaceCOLMAP', ['-i', str(dense), '-o', 'scene.mvs'], 37, '准备稠密表面重建')
    run('DensifyPointCloud', ['scene.mvs', '-o', 'scene_dense.mvs', '--resolution-level', '0',
                            '--max-resolution', str(options['image_size']),
                            '--max-threads', str(min(os.cpu_count() or 4, 8)), '--tower-mode', '0'],
                            40, '计算多视图深度与稠密点云（CPU，可能需要较长时间）')
    run('ReconstructMesh', ['scene_dense.mvs', '-o', 'scene_mesh.mvs', '--close-holes', '0',
                           '--smooth', '0', '--remove-spurious', '0', '--remove-spikes', '0'],
                            45, '从稠密点云提取三角网格')
    run('RefineMesh', ['scene_mesh.mvs', '-o', 'scene_refined.mvs', '--resolution-level', '0',
                      '--max-resolution', str(options['image_size']), '--decimate', '1', '--close-holes', '0',
                      '--max-threads', str(min(os.cpu_count() or 4, 8))],
                            48, '根据照片细化网格表面')
    mesh_path = job / 'scene_refined.ply'
    if not mesh_path.exists():
        raise ValueError('OpenMVS 未输出预期的网格，请检查日志和引擎版本。')
    mesh = trimesh.load(mesh_path, process=True)
    cloud = trimesh.load(job / 'scene_dense.ply', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('重建结果没有有效表面。')
    if not hasattr(cloud, 'colors') or len(cloud.colors) != len(cloud.vertices):
        raise ValueError('稠密点云缺少颜色，不能进行照片对应颜色的自动分体。')
    # Preserve all components; never silently remove small object details.
    nearest = cKDTree(cloud.vertices).query(mesh.vertices, workers=2)[1]
    mesh.visual.vertex_colors = np.asarray(cloud.colors)[nearest]
    target = job / 'source.ply'
    mesh.export(target)
    return target
