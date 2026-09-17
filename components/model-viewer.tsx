import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

type Props = { url?: string; exploded: number; wireframe: boolean; hidden: string[]; reset: number; unlit: boolean; upAxis?: 'y'|'z'; onPick?: (point: number[]) => void; marks?: (number[]|undefined)[]; view?:'photo'|'front'|'orbit'; photoPitch?:number;photoYaw?:number };
export function ModelViewer({ url, exploded, wireframe, hidden, reset, unlit, upAxis='z', onPick, marks=[],view='orbit',photoPitch=0,photoYaw=0 }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const parts = useRef<THREE.Mesh[]>([]);
  const resetView = useRef<() => void>(() => {});
  const pick = useRef(onPick); pick.current=onPick;
  const preset=useRef({view,photoPitch,photoYaw});preset.current={view,photoPitch,photoYaw};
  const markerGroup = useRef<THREE.Group|undefined>(undefined);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    setLoading(false);
    resetView.current = () => {};
    if (!container.current) return;
    const mount = container.current;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch { setError('浏览器未启用 WebGL，无法显示三维预览。文件仍可正常导出。'); return; }
    setError('');
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x12191c, 0);
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x718088, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(80, 120, 100); scene.add(key);
    const fill = new THREE.DirectionalLight(0x9dc9ff, 1.5);
    fill.position.set(-80, 30, -60); scene.add(fill);
    const grid = new THREE.GridHelper(220, 22, 0x3b5357, 0x29383d);
    scene.add(grid);
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 5000);
    camera.position.set(155, 125, 170);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.target.set(0, 38, 0);
    controls.minDistance = 5; controls.maxDistance = 1500;
    let alive = true;
    let model: THREE.Group | undefined;
    const disposeModel = (root: THREE.Object3D) => root.traverse(o => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        const ms = o.userData.previewMaterials || (Array.isArray(o.material) ? o.material : [o.material]);
        ms.forEach((m: THREE.Material) => { for (const v of Object.values(m)) if (v instanceof THREE.Texture) v.dispose(); m.dispose(); });
      }
    });
    if (url) {
      setLoading(true);
      new GLTFLoader().load(url, gltf => {
        if (!alive) { disposeModel(gltf.scene); return; }
        model = new THREE.Group();
        gltf.scene.rotation.x = upAxis==='z' ? -Math.PI / 2 : 0;
        model.add(gltf.scene); scene.add(model);
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        model.position.set(-center.x, -box.min.y + .1, -center.z);
        const max = Math.max(...box.getSize(new THREE.Vector3()).toArray());
        controls.minDistance=max*.15; controls.maxDistance=max*20;
        markerGroup.current=new THREE.Group(); gltf.scene.add(markerGroup.current);
        grid.scale.setScalar(Math.max(max / 100, .2));
        parts.current = [];
        gltf.scene.traverse(o => { if (o instanceof THREE.Mesh) {
          const original = Array.isArray(o.material) ? o.material : [o.material];
          const flat = original.map(m => new THREE.MeshBasicMaterial({
            color: (m as THREE.MeshStandardMaterial).color,
            map: (m as THREE.MeshStandardMaterial).map,
            vertexColors: !!o.geometry.getAttribute('color'), side: m.side,
          }));
          o.userData.litMaterial = o.material;
          o.userData.flatMaterial = Array.isArray(o.material) ? flat : flat[0];
          o.userData.previewMaterials = [...original, ...flat];
          o.geometry.computeBoundingBox();
          o.userData.originalPosition = o.position.clone();
          o.userData.direction = o.geometry.boundingBox!.getCenter(new THREE.Vector3()).sub(new THREE.Vector3(0, 0, max*.5)).normalize();
          parts.current.push(o);
        }});
        resetView.current = () => {
          controls.target.set(0, box.getSize(new THREE.Vector3()).y*.46, 0);
          const choice=preset.current;
          if(upAxis==='y'||choice.view==='front')camera.position.copy(controls.target).add(new THREE.Vector3(0,0,max*2.8));
          else if(choice.view==='photo'){
            const p=THREE.MathUtils.degToRad(choice.photoPitch),y=THREE.MathUtils.degToRad(choice.photoYaw);
            camera.position.copy(controls.target).addScaledVector(new THREE.Vector3(-Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p)),max*2.8);
          }else camera.position.set(max*1.55,max*1.25,max*1.75);
          controls.update();
        };
        resetView.current(); setLoading(false);
      }, undefined, () => { if (alive) { setLoading(false); setError('模型预览载入失败，可尝试重新选择任务。'); } });
    }
    let pressed: {x:number;y:number}|undefined;
    const pointerDown=(event:PointerEvent)=>{pressed={x:event.clientX,y:event.clientY};};
    const pointerUp=(event:PointerEvent)=>{
      if(!pick.current||!pressed||Math.hypot(event.clientX-pressed.x,event.clientY-pressed.y)>5)return;
      pressed=undefined;
      const rect=renderer.domElement.getBoundingClientRect();
      const ray=new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,1-(event.clientY-rect.top)/rect.height*2),camera);
      const hit=ray.intersectObjects(parts.current,false)[0];
      if(hit) pick.current(hit.object.worldToLocal(hit.point.clone()).toArray());
    };
    renderer.domElement.addEventListener('pointerdown',pointerDown);
    renderer.domElement.addEventListener('pointerup',pointerUp);
    const observer = new ResizeObserver(() => {
      const { width, height } = mount.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height); camera.aspect = width/height; camera.updateProjectionMatrix();
    });
    observer.observe(mount);
    let frame = 0;
    const animate = () => { if (!alive) return; frame=requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); };
    animate();
    return () => {
      alive=false; cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
      renderer.domElement.removeEventListener('pointerdown',pointerDown); renderer.domElement.removeEventListener('pointerup',pointerUp);
      markerGroup.current=undefined;
      if (model) disposeModel(model); grid.geometry.dispose();
      (grid.material as THREE.Material).dispose(); parts.current=[];
      renderer.dispose(); renderer.domElement.remove();
    };
  }, [url,upAxis]);
  useEffect(()=>{
    const group=markerGroup.current;if(!group)return;
    for(const child of [...group.children]){const marker=child as THREE.Mesh;marker.geometry.dispose();(marker.material as THREE.Material).dispose();group.remove(child);}
    const colors=[0xff7b72,0x66b5ff,0xbadf7e];
    marks.forEach((point,i)=>{if(!point)return;const marker=new THREE.Mesh(new THREE.SphereGeometry(.016,12,8),new THREE.MeshBasicMaterial({color:colors[i%3],depthTest:false}));marker.position.fromArray(point);marker.renderOrder=2;group.add(marker);});
  },[marks,loading]);
  useEffect(() => {
    parts.current.forEach((part, i) => {
      const original = part.userData.originalPosition as THREE.Vector3;
      const direction = part.userData.direction as THREE.Vector3;
      part.position.copy(original).addScaledVector(direction, exploded*.5);
      if (direction.lengthSq() < .1) part.position.x += (i+1)*exploded*.3;
      part.visible = !hidden.includes(part.name);
      part.material = unlit ? part.userData.flatMaterial : part.userData.litMaterial;
      const materials = Array.isArray(part.material) ? part.material : [part.material];
      materials.forEach(m => { if ('wireframe' in m) (m as THREE.MeshStandardMaterial).wireframe=wireframe; });
    });
  }, [exploded, wireframe, hidden, loading, unlit]);
  useEffect(() => { resetView.current(); }, [reset,view,photoPitch,photoYaw]);
  return <><div ref={container} className="three-canvas" aria-label="三维模型预览，拖动旋转、滚轮缩放" />
    {loading && <div className="viewer-message">载入三维模型…</div>}
    {error && <div className="viewer-message error" role="alert">{error}</div>}</>;
}
