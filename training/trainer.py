"""Shared on-policy training routines for full-parameter and LoRA policies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
from importlib.metadata import version
from runtime_config import THINKING_MODE

ROOT = Path(__file__).resolve().parent
INSTRUCTION = (
 'Solve this mathematics problem carefully.\n'
 'In your final response, present your reasoning in at least two nonempty steps. '
 'Start each step on a new line with Step 1:, Step 2:, and so on, numbered consecutively without gaps. '
 'Each step should contain one meaningful mathematical deduction; do not add empty or trivial steps just to increase the count. '
 'Include these numbered steps in the final response even if you also use a separate thinking section.\n'
 'After the steps, write exactly one final-answer line: Final Answer: \\boxed{answer}. '
 'Replace answer with your result. Use \\boxed{...} only for this final answer, not for intermediate values. '
 'Write nothing after the final-answer line.\n\nProblem:\n')
# Labels only: paths and identities are explicit in runtime_config.py.
MODELS = ('qwen4b', 'qwen17b')

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
 from sampler_worker import atomic_json
 atomic_json(path, value)

def render(tok, row):
 text = tok.apply_chat_template([{'role':'user','content':INSTRUCTION+row['question']}], tokenize=False, add_generation_prompt=True, enable_thinking=THINKING_MODE)
 ids = tok(text, add_special_tokens=False)['input_ids']
 assert tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
 assert 0 < len(ids) < 8192
 return ids

def schedule(rows, step, batch=4):
 order = list(range(len(rows)))
 random.Random(42+step//(len(rows)//batch)).shuffle(order)
 start = (step*batch) % len(rows)
 return [rows[i] for i in order[start:start+batch]]

def loss_terms(new_logp, old_logp, advantages, clip=.2):
 import torch
 difference = new_logp - old_logp
 if not bool(torch.isfinite(difference).all()):
  raise ValueError('Nonfinite actor/behavior log probability')
 ratio = difference.clamp(-20,20).exp()
 clipped = ratio.clamp(1-clip,1+clip)
 loss = -torch.minimum(ratio*advantages, clipped*advantages).sum()
 return loss, difference.detach(), (ratio.detach()-1).abs() > clip

def selected_logps(actor, prompt_ids, response_ids):
 import torch
 import torch.nn.functional as F
 from policy_runtime import forward_context
 ids = torch.tensor([prompt_ids+response_ids], device=next(actor.parameters()).device, dtype=torch.long)
 n = len(response_ids)
 with forward_context(actor):
  logits = actor(input_ids=ids[:,:-1], use_cache=False, logits_to_keep=n).logits[0]
 assert logits.shape[0] == n
 targets = ids[0,-n:]
 # FP32 normalization is deliberate; no sampled-token retokenization.
 return -F.cross_entropy(logits.float(), targets, reduction='none')

def load_actor(model, adapter=None, *, device='cuda', finetune_mode='full', checkpoint=None):
 import torch
 from transformers import AutoModelForCausalLM
 if finetune_mode not in ('full', 'lora'):
  raise ValueError('Choose full or lora fine-tuning')
 if finetune_mode == 'full':
  if adapter is not None:
   raise ValueError('Full training cannot initialize from a LoRA adapter')
  # Keep FP32 master parameters and AdamW state; FSDP casts compute to BF16.
  # In particular, do not round resumed FP32 master weights through BF16.
  actor = AutoModelForCausalLM.from_pretrained(checkpoint or model, torch_dtype=torch.float32,
              attn_implementation='sdpa', local_files_only=True).to(device)
  actor.requires_grad_(True)
 else:
  if checkpoint is not None:
   raise ValueError('LoRA training expects an adapter, not a full checkpoint')
  from peft import LoraConfig, PeftModel, get_peft_model
  base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16,
              attn_implementation='sdpa', local_files_only=True).to(device)
  if adapter:
   # PEFT otherwise chooses a globally available accelerator even when the
   # actor was explicitly loaded on CPU (for example MPS during CPU tests).
   actor = PeftModel.from_pretrained(base, adapter, is_trainable=True,
               autocast_adapter_dtype=True, torch_device=str(device))
  else:
   actor = get_peft_model(base, LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
  for name,p in actor.named_parameters():
   if p.requires_grad:
    assert 'lora_' in name
    p.data = p.data.float()
 actor.config.use_cache = False
 actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 return actor

def load_reference(model, *, device='cuda'):
 """Frozen initial encoder for full training; never updated or resumed from policy."""
 import torch
 from transformers import AutoModel
 reference = AutoModel.from_pretrained(model, torch_dtype=torch.bfloat16,
               attn_implementation='sdpa', local_files_only=True).to(device)
 reference.requires_grad_(False)
 reference.eval()
 reference.config.use_cache = False
 return reference

def save_adapter(actor, folder):
 folder = Path(folder)
 assert not folder.exists(), 'Adapters are immutable'
 temporary = folder.with_name(folder.name+'.writing')
 assert not temporary.exists()
 actor.save_pretrained(temporary, safe_serialization=True)
 temporary.rename(folder)
 write(folder.with_name(folder.name+'_manifest.json'),{p.name:digest(p) for p in folder.iterdir() if p.is_file()})
 return str(folder)

def prune_adapters(out, current_step):
 # Only this run's disposable synchronization copies; preserve latest two,
 # their external manifests and the final checkpoint.
 for old in Path(out).glob('adapter_*'):
  if re.fullmatch(r'adapter_[0-9]{3,}', old.name) is None:
   continue
  if int(old.name.split('_')[1]) < current_step-1:
   assert old.parent.resolve()==Path(out).resolve() and not old.is_symlink()
   shutil.rmtree(old)

def features(actor, prompt, ids, spans, *, include_pooled=False):
 import torch
 from contextlib import nullcontext
 from feature_pooling import pool_reference_hidden
 is_lora = hasattr(actor, 'disable_adapter')
 if not is_lora and any(p.requires_grad for p in actor.parameters()):
  raise ValueError('Full training requires a separate frozen reference encoder')
 with torch.no_grad(), actor.disable_adapter() if is_lora else nullcontext():
  x = torch.tensor([prompt+ids], dtype=torch.long, device=next(actor.parameters()).device)
  encoder = actor.get_base_model().model if is_lora else actor
  hidden = encoder(input_ids=x, use_cache=False).last_hidden_state[0,len(prompt):].float()
  return pool_reference_hidden(hidden, spans, include_pooled=include_pooled)

def annotate_groups(tok, rows, generated, actor=None, arm='grpo', *, format_coefficient=0., alpha=1.,
                    novelty_reference_scope='correct_only'):
 import torch
 from scoring import annotate
 from credit import standardize
 from experiment_credit import compute_group_credit
 from runtime_config import ARMS
 if arm not in ARMS:
  raise ValueError('Unknown comparison arm')
 if actor is None and arm != 'grpo':
  raise ValueError('STRIDE credit requires reference features')
 if format_coefficient != 0:
  raise ValueError('Structural rewards are disabled in this comparison')
 row_by_id = {r['id']:r for r in rows}
 groups = []
 for group in generated['rows']:
  records=[]; wholes=[]; steps=[]; pooled=[]; spans=[]; eligible=[]; truth=[]
  for sample in group['samples']:
   score = annotate(tok,group['prompt_token_ids'],sample['token_ids'],row_by_id[group['id']]['ground_truth'],sample['finish_reason'])
   records.append(dict(sample, score=score))
   truth.append(float(score['answer_correct']))
   sp=score['spans'];spans.append(sp);eligible.append(score['step_eligible'])
   if actor is not None:
    whole,step,single = features(actor,group['prompt_token_ids'],sample['token_ids'],sp,include_pooled=True)
    wholes.append(whole);steps.append(step);pooled.append(single)
  width=max(len(r['token_ids']) for r in records)
  # CAMD Torch has no CPU LAPACK. Run the unchanged credit kernels on the
  # learner GPU; reference extraction and its serialized values stay unchanged.
  credit_device = next(actor.parameters()).device if actor is not None and os.environ.get('STRIDE_RUNTIME') == 'rocm' else 'cpu'
  mask=torch.arange(width,device=credit_device)[None,:] < torch.tensor([len(r['token_ids']) for r in records],device=credit_device)[:,None]
  if actor is not None:
   credit=compute_group_credit(truth,steps,spans,eligible,torch.stack(wholes),mask,
                               pooled_step_embeddings=torch.stack(pooled),alpha=alpha,
                               novelty_reference_scope=novelty_reference_scope)
   advantage=credit['advantages'][arm];diagnostics=credit['diagnostics']
   feature_data={'whole':[w.tolist() for w in wholes], 'steps':[s.tolist() for s in steps],
                 'pooled_steps':[p.tolist() for p in pooled]}
  else:
   advantage=standardize(torch.tensor(truth))[:,None]*mask
   diagnostics={'correct_responses':sum(truth),'eligible_responses':sum(bool(e and t) for e,t in zip(eligible,truth))}
   feature_data=None
  groups.append(dict(id=group['id'],prompt_token_ids=group['prompt_token_ids'],max_tokens=group['max_tokens'],records=records,diagnostics=diagnostics,features=feature_data,advantages=advantage))
 return groups

def update(actor, optimizer, groups, *, force_signal=False):
 import torch
 actor.train();optimizer.zero_grad(set_to_none=True)
 device=next(actor.parameters()).device
 denominator=sum(len(r['token_ids']) for g in groups for r in g['records'])
 total_loss=0.;all_differences=[];clipped_count=0
 began=time.time()
 for group in groups:
  for index,record in enumerate(group['records']):
   new=selected_logps(actor,group['prompt_token_ids'],record['token_ids'])
   old=torch.tensor(record['logprobs'],device=device,dtype=torch.float32)
   advantage=group['advantages'][index,:len(old)].to(device)
   if force_signal: advantage=torch.ones_like(advantage)
   loss,diff,clipped=loss_terms(new,old,advantage)
   (loss/denominator).backward()
   total_loss+=float(loss.detach())/denominator
   all_differences.append(diff.cpu());clipped_count+=int(clipped.sum())
   del new,old,advantage,loss,diff,clipped
 gradient_norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.))
 if not __import__('math').isfinite(gradient_norm): raise ValueError('Nonfinite gradient')
 optimizer.step();actor.eval()
 differences=torch.cat(all_differences)
 return dict(loss=total_loss,gradient_norm=gradient_norm,response_tokens=denominator,update_seconds=time.time()-began,actor_behavior_abs_logp_mean=float(differences.abs().mean()),actor_behavior_abs_logp_p99=float(torch.quantile(differences.abs(),.99)),actor_behavior_abs_logp_max=float(differences.abs().max()),ratio_clip_fraction=clipped_count/denominator,peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)

def serialize_groups(groups):
 return [dict(g,advantages=g['advantages'].tolist()) for g in groups]

def request(tok,rows,identifier,adapter,adapter_id,seed,cap=8192,n=8,*,finetune_mode='lora'):
 if finetune_mode not in ('full', 'lora'):
  raise ValueError('Choose full or lora fine-tuning')
 payload = dict(request_id=identifier,prompts=[dict(id=r['id'],prompt_token_ids=render(tok,r),seed=seed+i) for i,r in enumerate(rows)],adapter_path=adapter if finetune_mode == 'lora' else None,adapter_id=adapter_id if finetune_mode == 'lora' else 0,n=n,max_tokens=cap,temperature=1.,context_limit=8192)
 if finetune_mode == 'full':
  payload.update(model_checkpoint=adapter, model_version=adapter_id)
 return payload


def main():
 """The obsolete fixed 32-question pilot entrypoint was intentionally removed."""
 raise SystemExit('Use torchrun with training/trainer_distributed.py smoke or training/trainer_resumable.py train; see README.md.')

if __name__ == '__main__':
 main()
