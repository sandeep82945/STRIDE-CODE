"""Read assigned-device mappings and test only scheduler-selected GPU masks."""
import json,os,subprocess,sys
from pathlib import Path
names=['SLURM_JOB_ID','SLURM_STEP_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']
root=Path('/path/to/cluster-user/system/stride_sample100_20260915/results')/('allocation-probe-'+os.environ['SLURM_JOB_ID'])
root.mkdir(exist_ok=False)
report={'environment':{k:os.environ.get(k) for k in names},'variants':{}}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('ASSIGNED_ENV',json.dumps(report['environment']),flush=True)
physical=os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
if physical is None or not physical.isdecimal():raise ValueError('Probe requires exactly one scheduler GPU')
probe="""import json,os,sys
from pathlib import Path
r={'environment':{k:os.environ.get(k) for k in ['ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}}
try:
 import torch
 r['available']=torch.cuda.is_available();r['count']=torch.cuda.device_count()
 if r['count']!=1:raise ValueError('Expected precisely one allocated visible GPU')
 x=torch.arange(16,device='cuda:0');torch.cuda.synchronize();r['sum']=int(x.sum());r['name']=torch.cuda.get_device_name(0);r['passed']=True
except Exception as e:r['passed']=False;r['error']=str(e)
Path(sys.argv[1]).write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r),flush=True)
"""
for label,rocr,drop in [('scheduler_normalized',os.environ.get('ROCR_VISIBLE_DEVICES'),False),('scheduler_no_ordinal',os.environ.get('ROCR_VISIBLE_DEVICES'),True),('step_physical',physical,True)]:
 env=os.environ.copy();env.update(ROCR_VISIBLE_DEVICES=rocr,CUDA_VISIBLE_DEVICES='0',HIP_VISIBLE_DEVICES='0')
 if drop:env.pop('GPU_DEVICE_ORDINAL',None)
 path=root/(label+'.json')
 subprocess.run([sys.executable,'-s','-c',probe,str(path)],env=env,check=False,timeout=90)
 if path.exists():
  report['variants'][label]=json.loads(path.read_text().replace('\\n','\n'))
 else:report['variants'][label]={'passed':False,'error':'No completion artifact'}
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PROBE_COMPLETE',json.dumps({k:v['passed'] for k,v in report['variants'].items()}),flush=True)
