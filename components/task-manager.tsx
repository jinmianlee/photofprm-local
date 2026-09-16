import { useState } from 'react';
import { api } from '@/lib/api';
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '@/components/ui/dialog';

export type Task = { id: string; kind: string; state: string; progress: number; message: string; storage_bytes?: number; updated_at?: number; started_at?: number };
type TaskFile = { name: string; bytes: number };
export const bytes = (value = 0) => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GB` : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(1)} MB` : `${(value / 1024).toFixed(1)} KB`;
const labels: Record<string, string> = { complete: '已完成', running: '正在生成', queued: '准备中', cancelled: '已取消', failed: '失败 / 中断' };
const kinds: Record<string, string> = { single: '单图生成', mesh: '网格分色', photos: '多图重建', demo: '分色测试', unknown: '未完成上传' };

export function TaskManager({ jobs, onOpen, onRefresh, onDeleted }: { jobs: Task[]; onOpen: (id: string) => void; onRefresh: () => void; onDeleted: (id: string) => void }) {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [deleting, setDeleting] = useState<Task>();
  const [working, setWorking] = useState('');
  const [error, setError] = useState('');
  const [files, setFiles] = useState<{ id: string; files: TaskFile[] }>();
  const act = async (task: Task, action: 'cancel' | 'delete') => {
    setWorking(task.id); setError('');
    try {
      await api(`/api/jobs/${task.id}${action === 'cancel' ? '/cancel' : ''}`, { method: action === 'cancel' ? 'POST' : 'DELETE' });
      if (action === 'delete') { onDeleted(task.id); setDeleting(undefined); if (files?.id === task.id) setFiles(undefined); }
      onRefresh();
    } catch (e) { setError((e as Error).message); }
    finally { setWorking(''); }
  };
  const inspect = async (task: Task) => {
    setError('');
    try { setFiles({ id: task.id, files: await api<TaskFile[]>(`/api/jobs/${task.id}/files`) }); }
    catch (e) { setError((e as Error).message); }
  };
  const shown = jobs.filter(j => (filter === 'all' || j.state === filter) && `${j.id} ${kinds[j.kind]}`.includes(query));
  return <section className="panel task-manager">
    <div className="panel-title"><div><span className="eyebrow">LOCAL LIBRARY</span><h2>任务与文件</h2><p className="hint">{jobs.length} 个任务 · 共 {bytes(jobs.reduce((total, j) => total + (j.storage_bytes || 0), 0))}。取消会保留文件；删除会清除该任务的照片、模型、日志和断点。</p></div><button className="secondary" onClick={onRefresh}>刷新</button></div>
    <div className="task-filters"><input className="number-input" aria-label="搜索任务" placeholder="搜索任务编号或类型" value={query} onChange={e => setQuery(e.target.value)} /><select className="number-input" aria-label="任务状态筛选" value={filter} onChange={e => setFilter(e.target.value)}><option value="all">全部状态</option>{Object.entries(labels).map(([state, label]) => <option key={state} value={state}>{label}</option>)}</select></div>
    {error && <p role="alert" className="error">{error}</p>}
    {!shown.length && <p className="parts-empty">没有符合条件的任务。</p>}
    {shown.map(task => { const active = ['running', 'queued'].includes(task.state); return <article className="task-row" key={task.id}>
      <div><strong>{kinds[task.kind] || task.kind} · {task.id.slice(0, 8)}</strong><p className="hint">{labels[task.state] || task.state}{active ? ` · ${task.progress || 0}%` : ''} · {bytes(task.storage_bytes)} · {new Date((task.started_at || task.updated_at || 0) * 1000).toLocaleString()}</p><p className="hint task-message">{task.message}</p></div>
      <div className="task-actions"><button className="secondary" onClick={() => onOpen(task.id)}>打开</button><button className="secondary" onClick={() => inspect(task)}>文件</button>{active ? <button className="secondary" disabled={working === task.id} onClick={() => act(task, 'cancel')}>取消生成</button> : <button className="secondary danger" disabled={working === task.id} onClick={() => { setError(''); setDeleting(task); }}>删除任务及文件</button>}</div>
    </article>; })}
    <Dialog open={!!deleting} onOpenChange={open => { if (!open && !working) setDeleting(undefined); }}><DialogContent><DialogTitle>删除任务 {deleting?.id.slice(0, 8)}？</DialogTitle><DialogDescription>将永久删除这个任务的全部照片、生成模型、断点和日志，约 {bytes(deleting?.storage_bytes)}。其他任务和共享模型权重不受影响。下载到其他文件夹的副本也会保留。</DialogDescription>{error && <p className="error" role="alert">{error}</p>}<button className="secondary danger" disabled={!!working} onClick={() => deleting && act(deleting, 'delete')}>{working ? '正在删除…' : '确认删除全部任务文件'}</button><button className="secondary" disabled={!!working} onClick={() => setDeleting(undefined)}>保留任务</button></DialogContent></Dialog>
    <Dialog open={!!files} onOpenChange={open => { if (!open) setFiles(undefined); }}><DialogContent className="log-dialog"><DialogTitle>任务文件 · {files?.id.slice(0, 8)}</DialogTitle><DialogDescription>文件保存在服务电脑。删除任务可以一次清理这些文件；已完成的模型可从“检查并导出”下载。</DialogDescription><div className="task-files">{files?.files.map(file => <div key={file.name}><span>{file.name}</span><span>{bytes(file.bytes)}</span></div>)}</div></DialogContent></Dialog>
  </section>;
}
