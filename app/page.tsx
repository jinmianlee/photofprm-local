import { useEffect, useRef, useState } from 'react';
import { Box, Camera, Upload, Layers3, Download, ArrowRight, Check, X, RotateCcw, ScanLine, Eye, EyeOff, Loader2, HardDrive, ImagePlus, Info, ChevronRight, FileBox, Settings2, TriangleAlert, Terminal } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { ModelViewer } from '@/components/model-viewer';
import { PhotoRegion, type Region } from '@/components/photo-region';
import { TaskManager } from '@/components/task-manager';
import { api as requestApi, ApiError } from '@/lib/api';

type Options = { size_mm: number; colors: number; pitch_mm: number; min_feature_mm: number; image_size: number; repair_small_holes: boolean; photo_pitch_deg: number; photo_yaw_deg: number; color_style: 'photo'|'flat'; photo_auto_align: boolean; mesh_resolution: number; palette_override: string[] };
type Part = { name: string; color: string; file: string; faces: number; watertight: boolean; volume_mm3: number; components: number; small_islands: number };
type Job = { storage_bytes?: number; updated_at?: number; id: string; state: string; progress: number; message: string; kind: string; can_resume?: boolean; can_refine?: boolean; raw_shape_available?: boolean; started_at?: number; elapsed_seconds?: number; source_available?: boolean; source_file?: string; capture?: { uploaded: number; registered: number }; report?: { options?: Partial<Options>; generation?: { flat_palette_rgb?: number[][]; color_method?: string }; final: { dimensions_mm: number[]; faces: number }; parts: Part[]; warnings: string[]; timings?: { phases: { phase: string; seconds: number; state: string }[] }; sampling: { pitch_mm: number; partition_volume_relative_error?: number } } };
type System = { network_mode?: string; mesh_ready: boolean; reconstruction_ready: boolean; backend: string; engines: Record<string, boolean>; single_photo?: { ready: boolean; installed: boolean; message: string; model: string; device: string; download_bytes: number; download_total: number; download_state: string } };
const initial: Options = { size_mm: 95, colors: 3, pitch_mm: .8, min_feature_mm: .8, image_size: 2400, repair_small_holes: false, photo_pitch_deg: 25, photo_yaw_deg: 7, color_style: 'flat', photo_auto_align: true, mesh_resolution: 383, palette_override: [] };
const rgbHex = (rgb: number[]) => '#'+rgb.map(v=>v.toString(16).padStart(2,'0')).join('');
const phaseNames: Record<string,string> = { foreground: '去除背景', shape_inference_and_extraction: '生成形状与提取网格', color_projection_and_base: '整理底面与上色', color_partition_and_export: '实体分色与导出' };
function duration(seconds: number) {
  const total = Math.max(0, Math.floor(seconds));
  return total >= 60 ? `${Math.floor(total/60)} 分 ${total%60} 秒` : `${total} 秒`;
}
const api = <T = Job,>(path: string, init?: RequestInit) => requestApi<T>(path, init);
function PhotoThumb({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [url, setUrl] = useState('');
  useEffect(() => { const url=URL.createObjectURL(file); setUrl(url); return () => URL.revokeObjectURL(url); }, [file]);
  return <div className="photo-thumb"><img src={url} alt={file.name} /><button onClick={onRemove} aria-label={'移除 '+file.name}><X size={13}/></button></div>;
}
function Choice({ value, onChange, items, label }: { value: string; onChange: (value: string) => void; items: [string,string][]; label: string }) {
  return <Select value={value} onValueChange={v => v && onChange(v)} items={Object.fromEntries(items)}>
    <SelectTrigger className="choice" aria-label={label}><SelectValue /></SelectTrigger>
    <SelectContent>{items.map(([v,t]) => <SelectItem key={v} value={v}>{t}</SelectItem>)}</SelectContent>
  </Select>;
}
export default function Home() {
  const [system, setSystem] = useState<System>();
  const [step, setStep] = useState(0);
  const [page, setPage] = useState<'workflow'|'tasks'>('workflow');
  const [needsPair, setNeedsPair] = useState(false);
  const [accessKey, setAccessKey] = useState('');
  const checkConnection = () => api<System>('/api/system').then(value => { setSystem(value); setNeedsPair(false); setError(''); }).catch(e => { setNeedsPair(e instanceof ApiError && e.status === 401); setError(e instanceof ApiError ? e.message : '无法连接服务电脑。请启动“启动应用.cmd”并保持运行；局域网请检查服务地址。'); });
  const pair = async () => {
    try { await api('/api/connect', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ key: accessKey }) }); setAccessKey(''); await checkConnection(); refreshHistory(); }
    catch(e) { setError((e as Error).message); }
  };
  const [options, setOptions] = useState<Options>(initial);
  const [kind, setKind] = useState('single');
  const [photos, setPhotos] = useState<File[]>([]);
  const [region, setRegion] = useState<Region | null>(null);
  const [mesh, setMesh] = useState<File>();
  const [job, setJob] = useState<Job>();
  const [history, setHistory] = useState<Job[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [uploadPercent, setUploadPercent] = useState(0);
  const [now, setNow] = useState(Date.now());
  const [wireframe, setWireframe] = useState(false);
  const [unlit, setUnlit] = useState(true);
  const [exploded, setExploded] = useState(0);
  const [hidden, setHidden] = useState<string[]>([]);
  const [reset, setReset] = useState(0);
  const [guide, setGuide] = useState(false);
  const [camera, setCamera] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [log, setLog] = useState<string>();
  const video = useRef<HTMLVideoElement>(null);
  const photoInput = useRef<HTMLInputElement>(null);
  const meshInput = useRef<HTMLInputElement>(null);
  const stream = useRef<MediaStream | undefined>(undefined);
  const xhr = useRef<XMLHttpRequest | undefined>(undefined);
  const selectionVersion = useRef(0);
  const clearResult = () => {
    selectionVersion.current += 1;
    setJob(undefined); setLog(undefined); setError('');
    localStorage.removeItem('photoform-job');
    window.history.replaceState(null, '', window.location.pathname);
  };
  const running = busy || !!job && ['queued','running'].includes(job.state);
  const done = job?.state === 'complete';
  const otherActive = history.find(h=>h.id!==job?.id && ['queued','running'].includes(h.state));
  const generationBlocked = running || !!otherActive;
  const hasInput = kind==='mesh' ? !!mesh : kind==='single' ? photos.length===1 : photos.length>=8;
  const elapsed = running && job?.started_at ? Math.max(0, now/1000-job.started_at) : job?.elapsed_seconds;
  const parts = job?.report?.parts || [];
  const materialPalette = job?.report?.generation?.flat_palette_rgb || [];
  const setMaterial = (index: number, value: string) => setOptions(o=>({...o,
    palette_override: materialPalette.map((rgb,i)=>i===index?value:o.palette_override[i]||rgbHex(rgb))}));
  const kindName = (value: string) => value==='single'?'单图生成':value==='photos'?'照片重建':value==='demo'?'分色测试模型':'网格分色';
  const fileUrl = (name: string) => `/api/jobs/${job?.id}/files/${name}`;
  const setOption = <K extends keyof Options>(k: K, v: Options[K]) => setOptions(o => ({ ...o, [k]: v }));
  const refreshHistory = () => { void api<Job[]>('/api/jobs').then(setHistory).catch(() => {}); };
  useEffect(() => {
    void checkConnection();
    const version = selectionVersion.current;
    api<Job[]>('/api/jobs').then((jobs: Job[]) => {
      setHistory(jobs);
      // Completed history is opened explicitly; it cannot represent a new upload.
      const requested = new URLSearchParams(window.location.search).get('job');
      const pick = requested && /^[0-9a-f]{32}$/.test(requested)
        ? jobs.find(j=>j.id===requested) : jobs.find(j=>['running','queued'].includes(j.state));
      if(pick) api('/api/jobs/'+pick.id).then(next => {
        if(selectionVersion.current === version) { setJob(next); setStep(next.state==='complete'?2:1); }
      }).catch(()=>{});
    }).catch(()=>{});
  }, []);
  useEffect(() => {
    const timer = setInterval(() => { void api<System>('/api/system').then(value=>{setSystem(value);setNeedsPair(false);}).catch(e=>{setSystem(undefined);if(e instanceof ApiError && e.status===401)setNeedsPair(true);}); refreshHistory(); }, 8000);
    return () => clearInterval(timer);
  }, [system?.single_photo?.ready]);
  useEffect(() => {
    if (!job) return;
    localStorage.setItem('photoform-job', job.id);
    window.history.replaceState(null, '', `${window.location.pathname}?job=${job.id}`);
    if (!['queued','running'].includes(job.state)) return;
    const version = selectionVersion.current;
    let stopped=false;
    const timer=setInterval(() => api('/api/jobs/'+job.id).then((next: Job) => { if(!stopped && selectionVersion.current === version) { setJob(next); if(next.state==='complete')setStep(2); if(!['queued','running'].includes(next.state)) refreshHistory(); } }).catch(() => { if(!stopped) setError('任务状态连接中断，请确认本地应用仍在运行。'); }), 1800);
    return () => { stopped=true; clearInterval(timer); };
  }, [job]);
  useEffect(() => { setHidden([]); setExploded(0); }, [job?.id]);
  useEffect(() => {
    if (job?.report?.options) setOptions({...initial, ...job.report.options,
      color_style: job.report.options.color_style || (job.report.generation?.flat_palette_rgb?'flat':'photo')});
  }, [job?.id, job?.state]);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);
  useEffect(() => {
    if (!camera) { stream.current?.getTracks().forEach(t=>t.stop()); return; }
    let cancelled=false; setCameraError('');
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 2560 }, height: { ideal: 1920 } }, audio: false })
      .then(s => { if(cancelled) { s.getTracks().forEach(t=>t.stop()); return; } stream.current=s; if(video.current) video.current.srcObject=s; })
      .catch(() => setCameraError('无法使用摄像头。请允许摄像头权限，或使用“添加照片”上传已拍摄的照片。'));
    if(!navigator.mediaDevices) setCameraError('此浏览器没有可用摄像头接口，请上传照片。');
    return () => { cancelled=true; stream.current?.getTracks().forEach(t=>t.stop()); };
  }, [camera]);
  const addPhotos = (files: File[]) => {
    clearResult();
    setRegion(null);
    // Material indices belong to the selected photo's palette.
    setOptions(o=>({...o,palette_override:[]}));
    const allowed=files.filter(f=>/\.(jpe?g|png|webp)$/i.test(f.name));
    if(allowed.length!==files.length) setError('已跳过不支持的文件；请使用 JPG、PNG、WebP。');
    setPhotos(old => kind==='single' ? allowed.slice(0,1) : [...old,...allowed].slice(0,300));
    if(kind==='single' && allowed.length>1) setError('单图模式已选取第一张照片。');
  };
  const takePhoto = () => {
    if(!video.current?.videoWidth) return;
    const canvas=document.createElement('canvas'); canvas.width=video.current.videoWidth; canvas.height=video.current.videoHeight;
    canvas.getContext('2d')?.drawImage(video.current,0,0);
    canvas.toBlob(blob=> { if(blob) addPhotos([new File([blob], `capture_${Date.now()}.jpg`, { type:'image/jpeg' })]); }, 'image/jpeg', .98);
  };
  const startJob = async (mode: 'upload'|'demo'|'reprocess'|'resume'|'refine') => {
    const previousJobId = job?.id;
    clearResult();
    setStep(1); setPage('workflow');
    setError(''); setBusy(true); setUploadPercent(0);
    try {
      let result: { id: string };
      if(mode==='upload') {
        const files=kind!=='mesh'?photos:mesh?[mesh]:[];
        if(!files.length) throw new Error('请先添加照片或模型。');
        if(kind==='photos' && files.length<8) throw new Error('完整立体重建至少需要 8 张不同角度照片，建议 60–150 张。');
        const data=new FormData(); data.append('kind',kind); data.append('options',JSON.stringify(options)); files.forEach(f=>data.append('files',f));
        if(kind==='single' && region) data.append('region',JSON.stringify(region));
        result=await new Promise((resolve,reject) => {
          const request=new XMLHttpRequest(); xhr.current=request;
          request.open('POST','/api/upload'); request.setRequestHeader('X-PhotoForm-Client','local');
          request.upload.onprogress=e=> { if(e.lengthComputable) setUploadPercent(Math.round(100*e.loaded/e.total)); };
          request.onerror=()=>reject(new Error('上传失败，请检查本地服务。'));
          request.onabort=()=>reject(new Error('上传已取消。'));
          request.onload=()=> { try { const r=JSON.parse(request.responseText); if(request.status<300) resolve(r); else reject(new Error(typeof r.detail==='string'?r.detail:'上传文件或参数无效。')); } catch { reject(new Error('服务返回异常，请查看本地服务日志。')); } };
          request.send(data);
        });
      } else result=await api(mode==='demo'?'/api/demo':`/api/jobs/${previousJobId}/${mode}`, { method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(options) });
      setJob(await api('/api/jobs/'+result.id)); refreshHistory();
    } catch(e) { setError((e as Error).message); }
    finally { setBusy(false); xhr.current=undefined; }
  };
  const cancel = async () => {
    if(busy) { xhr.current?.abort(); return; }
    try { setJob(await api(`/api/jobs/${job?.id}/cancel`,{method:'POST'})); refreshHistory(); } catch(e) { setError((e as Error).message); }
  };
  const selectJob = async (id: string) => {
    clearResult(); const version=selectionVersion.current;
    try { const next=await api('/api/jobs/'+id); if(selectionVersion.current===version) { setJob(next); setStep(next.state==='complete'?2:1); setPage('workflow'); if(next.can_refine)setKind('single');else if(['single','photos','mesh'].includes(next.kind))setKind(next.kind); } }
    catch(e) { if(selectionVersion.current===version) setError((e as Error).message); }
  };
  if(needsPair) return <main className="pair-screen"><form className="panel" onSubmit={e=>{e.preventDefault();void pair();}}><span className="eyebrow">PHOTOFORM LOCAL</span><h1>连接你的模型电脑</h1><p className="hint">输入服务电脑“启动局域网.cmd”窗口里显示的访问码。照片与模型由那台电脑处理和保存。</p><label htmlFor="access-key">局域网访问码</label><input autoFocus id="access-key" type="password" className="number-input" autoComplete="off" value={accessKey} onChange={e=>setAccessKey(e.target.value)}/>{error&&<p className="error">{error}</p>}<button className="primary full" disabled={!accessKey}>连接</button></form></main>;
  return <div className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-icon"><Box size={24}/></span><div>PhotoForm <span className="brand-tag">LOCAL</span><small>照片 · 实体 · 打印</small></div></div>
      <div className="header-right"><button className="secondary" onClick={()=>{setPage(page==='tasks'?'workflow':'tasks');refreshHistory();}}>{page==='tasks'?'返回建模':'任务与文件'}</button><span className="local-badge"><span/>本机处理</span><button className="quiet-button" onClick={()=>setGuide(true)}><Info size={16}/>拍摄与打印指南</button></div></header>
    <nav className="steps" aria-label="建模流程">{['导入照片 / 网格','设置与生成','检查并导出'].map((title,i)=><button aria-current={page==='workflow'&&step===i?'step':undefined} disabled={i===2&&!done||i===0&&running||i===1&&!hasInput&&!job} className={page==='workflow'&&step===i?'step active':'step'} key={title} onClick={()=>{setPage('workflow');setStep(i);}}><span>{i+1}</span>{title}{i<2&&<ChevronRight size={16}/>}</button>)}</nav>
    {!system&&<div className="connection-notice" role="alert">服务连接中断或正在启动。请在服务电脑运行启动器并保持运行。<button className="secondary" onClick={()=>void checkConnection()}>重新连接</button></div>}
    {error&&<div className="connection-notice error" role="alert">{error}<button className="quiet-button" onClick={()=>setError('')}>关闭提示</button></div>}
    {otherActive&&<div className="connection-notice">任务 {otherActive.id.slice(0,8)} 正在处理 · {otherActive.progress}%<button className="secondary" onClick={()=>void selectJob(otherActive.id)}>查看正在运行的任务</button></div>}
    {page==='tasks'&&<main className="library-workspace"><TaskManager jobs={history} onOpen={selectJob} onRefresh={refreshHistory} onDeleted={id=>{if(job?.id===id){clearResult();setStep(0);}refreshHistory();}}/></main>}
    <main className={'workspace wizard step-'+step} hidden={page!=='workflow'}>
      <div className="workflow-heading"><span className="eyebrow">STEP {step+1} / 3</span><h1>{['先选择建模素材','设置尺寸和颜色，开始生成','检查模型，带到切片软件'][step]}</h1><p className="hint">{['上传照片后再进入设置；已有结果可从右上角“任务与文件”打开。','每次只运行一个生成任务。切换到任务管理仍可查看、取消正在运行的任务。','旋转检查形状、部件和打印提示。需要调整时返回第二步，复用已有形状。'][step]}</p></div>
      <aside className="panel source-panel" hidden={step!==0}><div className="panel-title"><div><span className="eyebrow">01 / CAPTURE</span><h2>素材</h2></div><ImagePlus size={20}/></div>
        <Tabs value={kind} onValueChange={v=>{if(!running){clearResult();setRegion(null);setKind(String(v));if(v==='single')setPhotos(old=>old.slice(0,1));}}}><TabsList className="source-tabs"><TabsTrigger disabled={running} value="single">单张照片</TabsTrigger><TabsTrigger disabled={running} value="photos">多角度</TabsTrigger><TabsTrigger disabled={running} value="mesh">已有模型</TabsTrigger></TabsList>
          <TabsContent value={kind==='single'?'single':'photos'}><input ref={photoInput} type="file" multiple={kind!=='single'} accept="image/jpeg,image/png,image/webp" hidden onChange={e=>{addPhotos(Array.from(e.target.files||[]));e.target.value='';}}/>
            <button className="dropzone" disabled={running} onClick={()=>photoInput.current?.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!running)addPhotos(Array.from(e.dataTransfer.files));}}><span className="upload-mark"><Upload size={23}/></span><strong>{kind==='single'?'上传主体照片':'添加照片'}</strong><span>点击选择或拖放到这里</span><small>JPG、PNG、WebP · {kind==='single'?'一张清晰照片':'最多 300 张'}</small></button>
            <button className="secondary full" disabled={running} onClick={()=>setCamera(true)}><Camera size={17}/>使用摄像头拍摄</button>
            <div className="photo-meta"><span><b>{photos.length}</b> 张照片</span>{photos.length>0&&<button onClick={()=>{clearResult();setRegion(null);setPhotos([]);}} disabled={running}>清空</button>}</div>
            {photos.length>0&&!(kind==='single'&&region)&&<div className={kind==='single'?'single-reference':'photo-grid'}>{photos.map((f,i)=><PhotoThumb key={f.name+i} file={f} onRemove={()=>{if(!running){clearResult();setRegion(null);setPhotos(old=>old.filter((_,n)=>i!==n));}}}/>)}</div>}
            {kind==='single'&&photos[0]&&<PhotoRegion file={photos[0]} value={region} disabled={running} onChange={value=>{clearResult();setRegion(value);setOptions(o=>({...o,palette_override:[]}));}}/>}
            <div className="capture-tip"><ScanLine size={18}/><div><b>{kind==='single'?'主体清晰，轮廓完整':'环绕物体拍摄 2–3 圈'}</b><p>{kind==='single'?'适合从正面照片生成头胸像。侧面和背面由模型推断，正面颜色取自照片；可上传透明 PNG 改善轮廓。':'建议 60–150 张，覆盖顶部和底部。保持物体静止、光线柔和，相邻照片充分重叠。'}</p><button onClick={()=>setGuide(true)}>查看拍摄指南 <ArrowRight size={13}/></button></div></div>
          </TabsContent>
          <TabsContent value="mesh"><input ref={meshInput} hidden type="file" accept=".ply,.glb,.stl" onChange={e=>{clearResult();setMesh(e.target.files?.[0]);e.target.value='';}}/>
            <button className="dropzone" disabled={running} onClick={()=>meshInput.current?.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!running){clearResult();setMesh(e.dataTransfer.files[0]);}}}><FileBox size={30}/><strong>{mesh?mesh.name:'导入三角网格'}</strong><span>PLY / GLB / STL</span><small>单文件最大 200 MB</small></button>
            <p className="hint">彩色 PLY 或内嵌纹理 GLB 可自动提取颜色。STL 会生成单色实体。3DGS 点云请先在重建工具中提取网格表面。</p>
          </TabsContent>
        </Tabs>
        <div className="engine-status"><HardDrive size={17}/><div><strong>{kind==='single'?(system?.single_photo?.message||'连接本地服务…'):kind==='mesh'?(system?.mesh_ready?'网格处理已就绪':'检测网格环境…'):system?.reconstruction_ready?'照片重建引擎就绪':'照片重建引擎待安装'}</strong><span>{kind==='single'?`${system?.single_photo?.model||'Hunyuan3D-2mini-Turbo'} · ${system?.single_photo?.device||'本地 GPU'}`:kind==='mesh'?'封闭实体分色 · STL / 3MF':system?.backend||'检测运行环境'}</span></div><span className={'status-dot '+((kind==='single'?system?.single_photo?.ready:kind==='mesh'?system?.mesh_ready:system?.reconstruction_ready)?'ready':'')}/></div>
        {kind==='single'&&system?.single_photo&&!system.single_photo.ready&&<p className="hint">{system.single_photo.message}</p>}
        {kind==='photos'&&system&&!system.reconstruction_ready&&<p className="hint">运行应用文件夹中的「安装重建引擎.ps1」。已有网格的分色导出可单独使用。</p>}
        <button className="primary full" disabled={!hasInput||running} onClick={()=>setStep(1)}>下一步：设置与生成 <ArrowRight size={17}/></button>
      </aside>
      <section className="center-column" hidden={step===0}><div className="viewer">
        <div className="viewer-heading" hidden={step!==2}><span><span className="status-dot ready"/>三维预览</span><span>{done?(job?.kind==='demo'?'测试模型 · 非照片重建':'分色实体'):'毫米 · Z 轴向上'}</span></div>
        {step===2&&<ModelViewer url={done?fileUrl('preview.glb'):undefined} exploded={exploded} wireframe={wireframe} hidden={hidden} reset={reset} unlit={unlit}/>}
        {step===1&&!running&&<div className="viewer-empty"><span className="empty-icon"><Layers3 size={38}/></span><span className="eyebrow">准备生成</span><h1>{done?'调整已有模型':'确认生成设置'}</h1><p>{done?'可以重新分色或提取形状，原任务会保留。':`已选择${kind==='mesh'?'已有网格':kind==='single'&&region?'照片选区':kind==='single'?'单张照片':'多角度照片'}。设置好后点击“生成分色模型”。`}</p>{done&&<button className="primary" onClick={()=>setStep(2)}>返回模型与导出</button>}<small>生成耗时取决于硬件和参数；预览将在完成后显示。</small></div>}
        {running&&<output className="processing"><Loader2 className="spin" size={32}/><h2>{busy?'正在上传到本机':'正在处理模型'}</h2><p>{busy?`上传进度 ${uploadPercent}%` : job?.message}</p><div className="progress-track"><div style={{width:`${busy?uploadPercent:job?.progress||0}%`}}/></div><span>{busy?uploadPercent:job?.progress||0}% · {busy?'照片留在本机':elapsed!==undefined?`已用时 ${duration(elapsed)}`:'准备处理'}</span>{!busy&&<small>进度按处理阶段显示，各阶段耗时不同。</small>}<button className="quiet-button" onClick={cancel}><X size={14}/>取消任务</button></output>}
        <div className="viewer-bottom" hidden={step!==2}><span>拖动旋转 · 滚轮缩放 · 右键平移</span><div><button disabled={!done} className={wireframe?'on':''} onClick={()=>setWireframe(v=>!v)} title="线框" aria-label="切换线框"><ScanLine size={17}/></button><button disabled={!done} onClick={()=>setReset(v=>v+1)} title="复位视角" aria-label="复位视角"><RotateCcw size={17}/></button></div></div>
      </div>
      {(error||job?.state==='failed')&&<div className="error-box" role="alert"><TriangleAlert size={19}/><div><strong>需要处理</strong><p>{error||job?.message}</p>{job?.source_available&&<a href={fileUrl(job?.source_file || 'source.ply')}>下载原始彩色网格检查</a>}</div><button aria-label="关闭提示" onClick={()=>setError('')}><X size={16}/></button></div>}
      {job?.state==='cancelled'&&<div className="notice">{job.message}</div>}
      <div className="model-details" hidden={step!==2}><div className="detail-heading"><h2><Layers3 size={18}/>颜色与部件 <span>{parts.length||'—'}</span></h2><span className="hint">所有部件保持共同坐标</span></div>
        {!done?<div className="parts-empty">模型生成后，显示分色部件、网格检查和下载文件。</div>:<>
          <div className="model-stats"><div><small>整体尺寸</small><b>{job?.report?.final.dimensions_mm.join(' × ')} <em>mm</em></b></div><div><small>原始网格面数</small><b>{job?.report?.final.faces.toLocaleString()}</b></div><div><small>封闭实体检查</small><b className="good"><Check size={15}/>全部通过</b></div></div>
          <div className="switch-row"><label htmlFor="unlit">纯色预览 · 关闭显示光影</label><Switch id="unlit" checked={unlit} onCheckedChange={setUnlit}/></div>
          <div className="explode"><span>分体展开</span><Slider value={[exploded]} min={0} max={60} step={1} aria-label="分体展开距离" onValueChange={v=>setExploded(Array.isArray(v)?v[0]:v)}/><span>{exploded} mm</span></div>
          <div className="part-list">{parts.map((p,i)=><div className="part" key={p.name}><button aria-label={(hidden.includes(p.name)?'显示':'隐藏')+'部件'+(i+1)} onClick={()=>setHidden(v=>v.includes(p.name)?v.filter(n=>n!==p.name):[...v,p.name])}>{hidden.includes(p.name)?<EyeOff size={16}/>:<Eye size={16}/>}</button><span className="swatch" style={{background:p.color}}/><div><strong>部件 {String(i+1).padStart(2,'0')}</strong><small>{p.color.toUpperCase()} · {p.components} 个连通体{p.small_islands>0?` · ${p.small_islands} 处小孤岛`:''}</small></div><span className="part-check"><Check size={13}/>封闭</span><a href={fileUrl(p.file)} title={'下载部件 '+(i+1)} aria-label={'下载部件 '+(i+1)}><Download size={17}/></a></div>)}</div>
          <details className="quality-notes"><summary>打印前检查与精度说明</summary>{job?.report?.warnings.map(w=><p key={w}>{w}</p>)}<p>分色边界采样：{job?.report?.sampling.pitch_mm} mm。该数值不代表实物测量误差，也不是切片层高。</p></details>
          {job?.report?.timings&&<details className="quality-notes"><summary>本次处理耗时{elapsed!==undefined?` · ${duration(elapsed)}`:''}</summary>{job.report.timings.phases.filter(p=>phaseNames[p.phase]).map((p,i)=><p key={i}>{phaseNames[p.phase]}：{duration(p.seconds)}{p.state==='failed'?'（中断）':''}</p>)}<p>调整颜色、尺寸或颜色对齐时，使用“用当前参数重新分色”可复用已有形状。</p></details>}
        </>}
      </div>
      </section>
      <aside className="panel settings-panel" hidden={step===0}><div hidden={step!==1}><div className="panel-title"><div><span className="eyebrow">02 / FABRICATE</span><h2>生成设置</h2></div><Settings2 size={20}/></div>
        <fieldset disabled={running} className="settings-fields"><label className="field-label" htmlFor="size">模型最长边 <span>mm</span></label><input id="size" className="number-input" type="number" min={10} max={500} step={1} value={options.size_mm} onChange={e=>setOption('size_mm',Number(e.target.value))}/><p className="field-help">按实物尺寸校准，保持长宽高比例。</p>
          {kind==='single'&&<><div className="field-label">上色方式</div><Choice label="上色方式" value={options.color_style} onChange={v=>setOption('color_style',v as 'photo'|'flat')} items={[['flat','纯色打印 · 抑制阴影'],['photo','照片色彩 · 保留明暗']]}/><p className="field-help">纯色模式优先合并材质的明暗变化，减少碎色块。更改后可直接重新分色。</p><div className="field-label">形状细节</div><Choice label="形状细节" value={String(options.mesh_resolution)} onChange={v=>setOption('mesh_resolution',Number(v))} items={[['383','精细 · 383'],['255','标准 · 255']]}/><p className="field-help">精细模式提取更密的表面，耗时更长。不能补回照片缺失的结构。</p></>}
          <div className="field-label">颜色数量 <span>{options.colors} 色</span></div><Slider value={[options.colors]} min={1} max={8} step={1} aria-label="颜色数量" onValueChange={v=>setOptions(o=>({...o,colors:Array.isArray(v)?v[0]:v,palette_override:[]}))}/><div className="range-labels"><span>1</span><span>4</span><span>8</span></div><p className="field-help">按材料色分区。较少颜色通常能减少换色，但真实花纹也可能被合并。</p>
          {options.color_style==='flat'&&materialPalette.length===options.colors&&<details className="advanced" open><summary>自选材料色</summary><p className="field-help">点击色块选择耗材颜色，再重新分色。原照片的阴影不会改变这些色值。</p>{materialPalette.map((rgb,i)=><label className="switch-row" key={i}><span>材料色 {i+1}</span><input aria-label={`材料色 ${i+1}`} type="color" value={options.palette_override[i]||rgbHex(rgb)} onChange={e=>setMaterial(i,e.target.value)}/></label>)}<button className="quiet-button" onClick={()=>setOption('palette_override',[])}>恢复照片取色</button></details>}
          <label className="field-label" htmlFor="pitch">分色边界采样间距 <span>mm</span></label><input id="pitch" className="number-input" type="number" min={.15} max={3} step={.05} value={options.pitch_mm} onChange={e=>setOption('pitch_mm',Number(e.target.value))}/><p className="field-help">越小边界越细、占用内存越多。保留输入模型的外表面细节。</p>
          {kind==='photos'&&<><div className="field-label">照片处理尺寸</div><Choice label="照片处理尺寸" value={String(options.image_size)} onChange={v=>setOption('image_size',Number(v))} items={[[ '1600','快速 · 1600 px'],['2400','标准 · 2400 px'],['4000','精细 · 4000 px'],['6000','高分辨率 · 6000 px']]}/><p className="field-help">重建使用的照片最长边，原始照片保留。</p></>}
          {kind==='single'&&<p className="field-help">形状在本机生成，原始照片保留。没有照片信息的区域会被推断补全；请在导出前旋转检查。</p>}
          {kind==='single'&&<details className="advanced"><summary>照片颜色对齐</summary>{options.color_style==='flat'&&<div className="switch-row"><label htmlFor="auto-align">按轮廓估计视角</label><Switch id="auto-align" checked={options.photo_auto_align} onCheckedChange={v=>setOption('photo_auto_align',v)}/></div>}<p className="field-help">轮廓估计不等于五官识别。颜色错位时关闭自动估计，调整角度后重新分色。</p><label className="field-label" htmlFor="photo-pitch">上下视角 <span>°</span></label><input disabled={options.color_style==='flat'&&options.photo_auto_align} id="photo-pitch" className="number-input" type="number" min={-45} max={60} step={1} value={options.photo_pitch_deg} onChange={e=>setOption('photo_pitch_deg',Number(e.target.value))}/><label className="field-label" htmlFor="photo-yaw">左右视角 <span>°</span></label><input disabled={options.color_style==='flat'&&options.photo_auto_align} id="photo-yaw" className="number-input" type="number" min={-60} max={60} step={1} value={options.photo_yaw_deg} onChange={e=>setOption('photo_yaw_deg',Number(e.target.value))}/></details>}
          <details className="advanced"><summary>更多打印检查参数</summary><label className="field-label" htmlFor="feature">小孤岛提示阈值 <span>mm</span></label><input id="feature" className="number-input" type="number" value={options.min_feature_mm} min={.2} max={5} step={.1} onChange={e=>setOption('min_feature_mm',Number(e.target.value))}/><p className="field-help">提示容积小于阈值立方的孤岛；不代替薄壁厚度检测。</p><div className="switch-row"><label htmlFor="repair">修复三角形 / 四边形小孔</label><Switch id="repair" checked={options.repair_small_holes} onCheckedChange={v=>setOption('repair_small_holes',v)}/></div><p className="field-help">不会自动填补大范围缺失表面。</p></details>
        </fieldset>
        <button className="primary full generate" disabled={generationBlocked||!system?.mesh_ready||(kind==='single'?!system?.single_photo?.ready||photos.length!==1:kind==='photos'?!system?.reconstruction_ready||photos.length<8:!mesh)} onClick={()=>startJob('upload')}>{running?<Loader2 className="spin" size={18}/>:<Box size={18}/>}生成{kind==='single'&&region?'选区':'分色'}模型 <ArrowRight size={17}/></button>
        {job?.source_available&&!generationBlocked&&<button className="secondary full" onClick={()=>startJob('reprocess')}>用当前参数重新分色</button>}
        {job?.can_refine&&!generationBlocked&&<button className="secondary full" onClick={()=>startJob('refine')}>按 {options.mesh_resolution} 精度重新提取形状</button>}
        {job?.can_resume&&!generationBlocked&&<button className="secondary full" onClick={()=>startJob('resume')}>{job.raw_shape_available?'复用已生成形状，继续上色与导出':'从已保存的步骤继续'}</button>}
        <button className="quiet-button" disabled={running} onClick={()=>setStep(0)}>返回素材</button></div>
        <div className="export-block" hidden={step!==2}><span className="eyebrow">03 / EXPORT</span><h3>带到切片软件</h3><p className="hint">分色 STL 不携带颜色。优先使用 3MF 保留部件和显示颜色。</p>
          <a className={'export-link '+(!done?'disabled':'')} href={done?fileUrl('colored.3mf'):undefined} aria-disabled={!done}><span><b>彩色装配体</b><small>3MF · 保留位置与颜色</small></span><Download size={19}/></a>
          <a className={'export-link '+(!done?'disabled':'')} href={done?fileUrl('print_bundle.zip'):undefined} aria-disabled={!done}><span><b>完整打印文件包</b><small>分色 STL + 3MF + 检查报告</small></span><Download size={19}/></a>
          {done&&<a className="text-link" href={fileUrl('combined.stl')}>下载完整单色 STL <ArrowRight size={13}/></a>}
          <button className="secondary full" onClick={()=>setStep(1)}>返回设置 / 重新分色</button>
          <button className="quiet-button" onClick={()=>{clearResult();setPhotos([]);setMesh(undefined);setRegion(null);setOptions(initial);setStep(0);}}>开始新的模型</button>
        </div>
        {job&&<button className="quiet-button log-button" onClick={()=>api<{ text: string }>(`/api/jobs/${job.id}/log`).then(r=>setLog(r.text)).catch(e=>setError(e.message))}><Terminal size={15}/>查看任务日志</button>}
      </aside>
    </main>
    <footer><span><HardDrive size={13}/>照片与模型保存在本机 · 不调用云端生成服务</span><span>PhotoForm 0.2 · 本地单图生成 / 实体分色</span></footer>
    <Dialog open={guide} onOpenChange={setGuide}><DialogContent className="guide-dialog"><DialogTitle>拍得完整，才能打印得清楚</DialogTitle><DialogDescription>从一张照片生成头胸像，或通过多角度拍照扫描实物。</DialogDescription><ol><li><b>单张照片。</b>上传清晰的正面照片，尽量包含完整轮廓。生成后检查眼鼻和背面，必要时调整照片颜色对齐角度，再重新分色；无需重复推理。</li><li><b>多角度覆盖。</b>建议拍摄 60–150 张，分高、中、低三圈，邻近视角重叠约 70–80%。底部、凹槽和遮挡位置也要拍到。</li><li><b>让物体静止。</b>围绕物体移动相机，避免反光、透明、纯色物体和硬阴影。转动物体时，背景必须排除，否则会干扰相机定位。</li><li><b>精度由采集决定。</b>保持清晰对焦和一致曝光，物体尽量占满画面。单张照片不能可靠恢复背面；照片重建也不保证计量精度。</li><li><b>校准尺寸和检查缺口。</b>用尺测量实物最长尺寸。导出前检查孔洞、细杆和底部。应用只允许通过封闭实体检查的结果导出。</li><li><b>在切片软件中分配耗材。</b>优先打开 3MF。导入 STL 时同时选择全部分色部件，并作为一个对象的多个部件导入，保留共同坐标。</li><li><b>检查每一层。</b>确认薄壁、孤岛、悬垂、支撑及耗材槽位。分色部件没有装配间隙，更适合多材料打印；分别打印后拼装需要额外设计。</li></ol><p className="hint">3DGS/2DGS 需要先提取三角网格。本应用可接收这种流程导出的彩色 PLY / GLB；本机照片重建使用 COLMAP + OpenMVS。</p></DialogContent></Dialog>
    <Dialog open={camera} onOpenChange={setCamera}><DialogContent className="camera-dialog"><DialogTitle>{kind==='single'?'拍摄主体照片':'拍摄多角度照片'}</DialogTitle><DialogDescription>{kind==='single'?'拍摄一张主体清晰、轮廓完整的照片；重新拍摄会替换当前选择。':`每拍一张，缓慢移动相机到下一个角度。已添加 ${photos.length} 张。`}</DialogDescription><video ref={video} autoPlay playsInline muted/>{cameraError&&<p className="error">{cameraError}</p>}<button className="primary" onClick={takePhoto} disabled={!!cameraError}><Camera size={18}/>拍摄并添加</button><button className="secondary" onClick={()=>setCamera(false)}>完成拍摄</button></DialogContent></Dialog>
    <Dialog open={log!==undefined} onOpenChange={v=>{if(!v)setLog(undefined);}}><DialogContent className="log-dialog"><DialogTitle>任务日志</DialogTitle><DialogDescription>显示最近的处理输出，完整日志保存在任务目录。</DialogDescription><pre>{log||'暂时没有日志。'}</pre></DialogContent></Dialog>
  </div>;
}
