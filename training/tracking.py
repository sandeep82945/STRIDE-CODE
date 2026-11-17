"""Training W&B lifecycle: scalar metrics and explicit, non-secret metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid


def start_tracking(args, out, *, world_size, training_questions, updates):
    if args.wandb_mode == 'disabled':
        return None
    key = os.environ.get('WANDB_API_KEY', '').strip()
    if args.wandb_mode == 'online' and not key:
        raise ValueError('Set WANDB_API_KEY in .env or the environment before online training')
    metadata = {**vars(args), 'world_size': world_size, 'K': args.rollouts,
                'context_limit': 8192, 'updates': updates,
                'training_questions': training_questions,
                'loss_normalization': 'global_real_response_tokens'}
    if key and key in json.dumps(metadata, default=str):
        raise ValueError('W&B credentials must not appear in run metadata')
    # Never upload repository source, .env, models, or captured console output.
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    os.environ['WANDB_CONSOLE'] = 'off'
    run = None
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group,
                         name=f'{args.model}-{args.finetune_mode}-{args.mode}-{args.arm}-seed{args.seed}',
                         id=uuid.uuid4().hex, resume='never', job_type=args.mode,
                         config=metadata, dir=str(out), mode=args.wandb_mode,
                         save_code=False)
        if run is None:
            raise RuntimeError('W&B did not create a run')
        from sampler_worker import atomic_json
        atomic_json(Path(out) / 'wandb.json', dict(id=run.id, project=run.project,
                    entity=run.entity, url=run.url, mode=args.wandb_mode))
    except Exception:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise RuntimeError('W&B initialization failed; check your key, project access and network') from None
    return run


def setup_observers(args, rows, out, dist, *, updates):
    """Prepare held-out jobs and W&B once; all ranks see startup failures."""
    from sampler_worker import atomic_json
    from validation_jobs import prepare_validation, ValidationScheduler
    run, scheduler = None, None
    status = [None]
    if dist.get_rank() == 0:
        try:
            plan = prepare_validation(args, rows)
            atomic_json(Path(out) / 'validation_plan.json', plan)
            scheduler = ValidationScheduler(args, plan)
            run = start_tracking(args, out, world_size=dist.get_world_size(),
                                 training_questions=len(rows), updates=updates)
            status[0] = {'ok': True}
        except Exception as exc:
            key = os.environ.get('WANDB_API_KEY', '')
            message = str(exc)
            if key:
                message = message.replace(key, '[redacted]')
            status[0] = {'ok': False, 'error': message}
    dist.broadcast_object_list(status, src=0)
    if not status[0]['ok']:
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError('Validation/tracking setup failed: ' + status[0]['error'])
    return run, scheduler


def submit_validation(scheduler, run, *, completed_step, checkpoint, per_epoch):
    """Record scheduler status separately from evaluation results."""
    result = scheduler.submit_checkpoint(completed_step, checkpoint, per_epoch)
    if result is not None:
        print(json.dumps({'validation_submission': result}), flush=True)
        if run is not None:
            summary = scheduler.summary()
            run.summary['validation_jobs/submitted'] = summary['submitted']
            run.summary['validation_jobs/submission_failures'] = summary['submission_failures']
    return result
