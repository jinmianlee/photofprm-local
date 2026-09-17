import {useState} from 'react';
import {ModelViewer} from './model-viewer';
import {Dialog,DialogContent,DialogTitle,DialogDescription} from './ui/dialog';

export type Landmark={model:number[];photo:number[]};
const names=['画面左眼中心','画面右眼中心','鼻尖'];
export function PhotoCalibration({jobId,value,onApply,onClose}:{jobId:string;value:Landmark[];onApply:(points:Landmark[])=>void;onClose:()=>void}){
  const [index,setIndex]=useState(0);
  const [photos,setPhotos]=useState<(number[]|undefined)[]>(value.map(p=>p.photo));
  const [models,setModels]=useState<(number[]|undefined)[]>(value.map(p=>p.model));
  const complete=names.every((_,i)=>photos[i]&&models[i]);
  return <Dialog open onOpenChange={open=>!open&&onClose()}><DialogContent className="calibration-dialog">
    <DialogTitle>校准五官颜色位置</DialogTitle><DialogDescription>依次在照片和模型上点击同一个位置。先标两眼，再标鼻尖；可旋转模型，轻点选取。校准后重新分色，保留原任务。</DialogDescription>
    <div className="calibration-tabs">{names.map((name,i)=><button className={index===i?'active':''} key={name} onClick={()=>setIndex(i)}>{i+1}. {name} {photos[i]&&models[i]?'✓':''}</button>)}</div>
    <p className="field-help">当前：{names[index]} · 照片{photos[index]?'已标记':'待点击'} · 模型{models[index]?'已标记':'待点击'}</p>
    <div className="calibration-columns"><div><div className="field-label">原始照片</div><div className="calibration-photo">
      <img src={`/api/jobs/${jobId}/files/foreground.png`} alt="点击照片中的两眼中心和鼻尖" onClick={e=>{const r=e.currentTarget.getBoundingClientRect();const point=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];setPhotos(p=>{const next=[...p];next[index]=point;return next;});}}/>
      {photos.map((p,i)=>p&&<span key={i} className={`calibration-dot dot-${i}`} style={{left:`${p[0]*100}%`,top:`${p[1]*100}%`}}>{i+1}</span>)}
    </div></div><div><div className="field-label">实际生成形状</div><div className="calibration-model">
      <ModelViewer url={`/api/jobs/${jobId}/files/generated_raw.glb`} upAxis="y" unlit={false} exploded={0} wireframe={false} hidden={[]} reset={0} marks={models} onPick={point=>setModels(p=>{const next=[...p];next[index]=point;return next;})}/>
    </div></div></div>
    <div className="calibration-actions"><button className="quiet-button" onClick={()=>{setPhotos([]);setModels([]);setIndex(0);}}>清除标记</button><button className="secondary" onClick={onClose}>返回</button><button className="primary" disabled={!complete} onClick={()=>{onApply(names.map((_,i)=>({model:models[i]!,photo:photos[i]!})));onClose();}}>应用校准点</button></div>
  </DialogContent></Dialog>;
}
