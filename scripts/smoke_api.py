"""Exercise actual local HTTP upload, reprocessing, download and cancellation."""
from pathlib import Path
import json
import time
import zipfile
import io
import httpx

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {'X-PhotoForm-Client': 'local'}


def wait(client, job_id):
    for _ in range(60):
        response = client.get('/api/jobs/'+job_id)
        response.raise_for_status()
        value = response.json()
        if value['state'] in ('complete','failed','cancelled'):
            return value
        time.sleep(.5)
    raise RuntimeError('Task exceeded smoke-test time allowance')


if __name__ == '__main__':
    with httpx.Client(base_url='http://127.0.0.1:8765',headers=HEADERS,timeout=60,trust_env=False) as client:
        demo_id=(ROOT/'data'/'smoke-job.txt').read_text().strip()
        response=client.get('/api/jobs/'+demo_id+'/files/source.ply')
        response.raise_for_status()
        upload=client.post('/api/upload', data={'kind':'mesh','options':json.dumps({'size_mm':40,'colors':3,'pitch_mm':1})},
                           files={'files':('my-model.ply',response.content,'application/octet-stream')})
        upload.raise_for_status()
        imported=wait(client,upload.json()['id'])
        assert imported['state']=='complete', imported
        bundle=client.get('/api/jobs/'+imported['id']+'/files/print_bundle.zip')
        bundle.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert 'colored.3mf' in archive.namelist()
        reprocess=client.post('/api/jobs/'+imported['id']+'/reprocess',json={'size_mm':30,'colors':1,'pitch_mm':1})
        reprocess.raise_for_status()
        regenerated=wait(client,reprocess.json()['id'])
        assert regenerated['state']=='complete' and len(regenerated['report']['parts'])==1
        assert max(regenerated['report']['final']['dimensions_mm'])==30
        running=client.post('/api/demo',json={'size_mm':100,'colors':4,'pitch_mm':.8})
        running.raise_for_status()
        cancelled=client.post('/api/jobs/'+running.json()['id']+'/cancel')
        cancelled.raise_for_status()
        assert cancelled.json()['state']=='cancelled', cancelled.text
        final={'upload_job': imported['id'], 'reprocess_job':regenerated['id'], 'cancel_job':running.json()['id'],
               'upload':'passed','reprocess':'passed','download':'passed','cancel':'passed'}
        (ROOT/'data'/'api-smoke-report.json').write_text(json.dumps(final,indent=2),'utf-8')
        print(json.dumps(final,indent=2))
