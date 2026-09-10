import { useEffect, useRef, useState } from 'react';
import type { PointerEvent } from 'react';

export type Region = { x0: number; y0: number; x1: number; y1: number };
type Props = { file: File; value: Region | null; onChange: (region: Region | null) => void; disabled: boolean };
const full: Region = { x0: 0, y0: 0, x1: 1, y1: 1 };
const clamp = (v: number) => Math.min(1, Math.max(0, v));

export function PhotoRegion({ file, value, onChange, disabled }: Props) {
  const [url, setUrl] = useState('');
  const [size, setSize] = useState([0, 0]);
  const [draft, setDraft] = useState<Region | null>(null);
  const start = useRef<{ x: number; y: number; pointer: number } | null>(null);
  const area = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const next = URL.createObjectURL(file); setUrl(next); setSize([0, 0]); setDraft(null); start.current = null;
    return () => URL.revokeObjectURL(next);
  }, [file]);
  const point = (e: PointerEvent<HTMLDivElement>) => {
    const bounds = area.current!.getBoundingClientRect();
    return { x: clamp((e.clientX-bounds.left)/bounds.width), y: clamp((e.clientY-bounds.top)/bounds.height) };
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const origin = start.current;
    if (!origin || origin.pointer !== e.pointerId) return null;
    const p = point(e);
    const next = { x0: Math.min(p.x, origin.x), y0: Math.min(p.y, origin.y), x1: Math.max(p.x, origin.x), y1: Math.max(p.y, origin.y) };
    setDraft(next); return next;
  };
  const shown = draft || value || full;
  const pixels = [Math.round(shown.x1*size[0])-Math.round(shown.x0*size[0]), Math.round(shown.y1*size[1])-Math.round(shown.y0*size[1])];
  return <div className="region-editor">
    <div className="region-toggle"><button disabled={disabled} aria-pressed={!value} onClick={() => onChange(null)}>整个主体</button><button disabled={disabled} aria-pressed={!!value} onClick={() => onChange(value || full)}>框选部位</button></div>
    {value && <>
      <p>拖出选框，或调整下方边界。请保留完整头部与少量颈部。</p>
      <div className="region-frame" ref={area} style={{ touchAction: disabled ? 'auto' : 'none' }}
        onPointerDown={e => { if (disabled || e.button !== 0) return; e.preventDefault(); start.current = { ...point(e), pointer: e.pointerId }; e.currentTarget.setPointerCapture(e.pointerId); }}
        onPointerMove={move}
        onPointerUp={e => { const next = move(e); if (next && next.x1-next.x0 > .01 && next.y1-next.y0 > .01) onChange(next); start.current = null; setDraft(null); if(e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId); }}
        onPointerCancel={() => { start.current = null; setDraft(null); }}>
        <img src={url} alt="选择需要独立生成的部位" draggable={false} onLoad={e => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}/>
        <div className="region-box" style={{ left: `${shown.x0*100}%`, top: `${shown.y0*100}%`, width: `${(shown.x1-shown.x0)*100}%`, height: `${(shown.y1-shown.y0)*100}%` }}/>
      </div>
      <fieldset disabled={disabled} className="region-numbers">
        {([['x0','左'],['y0','上'],['x1','右'],['y1','下']] as const).map(([key, label]) => <label key={key}>{label}边界 %<input type="number" min={0} max={100} step={.1} value={Math.round(value[key]*1000)/10} onChange={e => { const n=e.currentTarget.valueAsNumber; if(Number.isFinite(n)) onChange({ ...value, [key]: clamp(n/100) }); }}/></label>)}
      </fieldset>
      <p className={size[0] && Math.min(...pixels)<256 ? 'error' : ''}>{pixels[0]} × {pixels[1]} 像素 · 短边至少 256 像素</p>
      <p>只生成选区的独立模型。不会自动装回整身；更多细节不代表相似度一定更高。</p>
    </>}
  </div>;
}
