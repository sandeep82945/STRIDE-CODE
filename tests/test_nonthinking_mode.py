"""Pinned native nonthinking prompts and incompatible-thinking resume rejection."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runtime_config as config
import trainer
import trainer_resumable
import resumable_checkpoint as checkpoints
from tests.test_training_config import arguments
from tests.test_hf_export import make_checkpoint


class NonthinkingModeTests(unittest.TestCase):
    def test_mode_is_fixed_recorded_and_fresh_initialization(self):
        args = arguments('train', '--finetune-mode', 'lora', '--model', 'qwen17b')
        config.validate_options(args, resumable=True)
        self.assertIs(args.thinking_mode, False)
        with patch('trainer_resumable.version', return_value='fixture'):
            contract, resume, start, steps = trainer_resumable.initial_training_state(args, 4, 2048, {})
        self.assertIs(contract['thinking_mode'], False)
        self.assertEqual(contract['initialization'], 'base_model_fresh_lora')
        self.assertEqual((resume, start, steps), (None, 0, 128))
        args.thinking_mode = True
        with self.assertRaisesRegex(ValueError, 'thinking_mode=False'):
            config.validate_options(args, resumable=True)

    def test_old_missing_or_true_mode_checkpoints_cannot_resume(self):
        args = arguments('train', '--finetune-mode', 'lora')
        config.validate_options(args, resumable=True)
        contract = config.scientific_contract(args, world_size=4, training_questions=2048, identity={})
        for old_mode in ('absent', True):
            old = copy.deepcopy(contract)
            if old_mode == 'absent':
                old.pop('thinking_mode')
            else:
                old['thinking_mode'] = old_mode
            with self.subTest(mode=old_mode), tempfile.TemporaryDirectory() as directory:
                run = Path(directory) / 'run'
                run.mkdir()
                make_checkpoint(run, 1, old, Path(directory) / 'base')
                with self.assertRaisesRegex(ValueError, 'Training contract changed'):
                    checkpoints.inspect_checkpoint(run / 'state_001', contract, 4)

    def test_epoch_extension_cannot_change_thinking_mode(self):
        args = arguments('train', '--finetune-mode', 'lora')
        config.validate_options(args, resumable=True)
        with patch('trainer_resumable.version', return_value='fixture'):
            previous = trainer_resumable.initial_training_state(args, 4, 2048, {})[0]
            previous['thinking_mode'] = True
            args.resume_from, args.allow_epoch_extension, args.epochs = '/old/state_128', True, 6
            with patch('trainer_resumable.checkpoints.inspect_checkpoint',
                       return_value=dict(contract=previous, completed_step=128)):
                with self.assertRaisesRegex(ValueError, 'only epochs'):
                    trainer_resumable.initial_training_state(args, 4, 2048, {})

    @unittest.skipUnless(os.environ.get('STRIDE_TEST_TOKENIZER_PATH'), 'Pinned local Qwen tokenizer required')
    def test_actual_tokenizer_training_and_shared_request_use_explicit_false(self):
        from transformers import AutoTokenizer
        from scoring import annotate
        from sampler_worker import validate_request
        tokenizer = AutoTokenizer.from_pretrained(os.environ['STRIDE_TEST_TOKENIZER_PATH'], local_files_only=True)
        row = dict(id='mode-proof', question='What is 1 + 1?', ground_truth=2)
        rendered = trainer.render(tokenizer, row)
        messages = [{'role': 'user', 'content': trainer.INSTRUCTION + row['question']}]
        off = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        on = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        decoded = tokenizer.decode(rendered, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.assertEqual(decoded, off)
        self.assertNotEqual(decoded, on)
        self.assertTrue(decoded.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertEqual(trainer.render(tokenizer, dict(row, ground_truth=999)), rendered)
        request = trainer.request(tokenizer, [row], 'off-mode', None, 0, 42)
        self.assertEqual(request['prompts'][0]['prompt_token_ids'], rendered)
        self.assertEqual(validate_request(request)['prompts'][0]['max_tokens'], 8192 - len(rendered))
        response = 'Step 1: Start with the number 1.\nStep 2: Adding 1 gives 2.\nFinal Answer: \\boxed{2}'
        ids = tokenizer(response, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        score = annotate(tokenizer, rendered, ids, 2, 'stop')
        self.assertTrue(score['answer_correct'])
        self.assertTrue(score['step_eligible'])
        self.assertEqual(score['region']['reason'], 'plain_response')
        truncated = annotate(tokenizer, rendered, ids, 2, 'length')
        self.assertFalse(truncated['answer_correct'])
        self.assertFalse(truncated['step_eligible'])
