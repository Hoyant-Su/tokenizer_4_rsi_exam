import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.language_model.batching import collator, length_bucketed, remaining_batches
from data.language_model.corpus import on_device
from data.language_model.storage import StoredWindows, run as prepare_tokens
from eval.language_model.prediction import evaluate
from model.language_model.causal import build, objective
from model.language_model.tokenization import load as load_tokenizer
from train.language_model.optimizers import build as build_optimizer
from train.language_model.schedule import warmup_cosine
from train.language_model.state import restore, save


def initialize(config):
    options = config['execution']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = options['cublas_workspace_config']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(options['num_threads'])
    torch.use_deterministic_algorithms(options['deterministic_algorithms'])
    torch.set_float32_matmul_precision(options['matmul_precision'])
    torch.backends.cudnn.benchmark = options['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = options['allow_tf32']
    torch.backends.cudnn.allow_tf32 = options['allow_tf32']



def run(config, stage):
    if stage == 'prepare_tokens':
        return prepare_tokens(config, stage)
    initialize(config)
    root = Path(config['output']['root'])
    root.mkdir(parents=True, exist_ok=True)
    (root / f'{stage}_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    tokenizer = load_tokenizer(config['tokenizer'])
    model = build(config, tokenizer.get_vocab_size())
    optimizer = build_optimizer(model, config)
    settings = config['training']
    progress = {'step': 0, 'epoch': 0, 'cursor': 0, 'samples': 0, 'tokens': 0,
                'input_positions': 0, 'training_seconds': 0.0, 'stage': stage}
    if stage == 'test':
        checkpoint = json.loads(Path(config['evaluation']['checkpoint_index']).read_text())
        progress = restore(model, optimizer, checkpoint['path'], config)
    elif settings['resume_path'] is not None:
        progress = restore(model, optimizer, settings['resume_path'], config)
    split = config['corpus']['test_split'] if stage == 'test' else config['corpus']['train_split']
    dataset = StoredWindows(config, split)
    indices = np.arange(len(dataset))
    metadata = {'parameters': sum(p.numel() for p in model.parameters()),
                'vocabulary_size': tokenizer.get_vocab_size(), 'seed': config['seed'],
                'source_records': len(dataset.records), 'selected_windows': len(dataset),
                'selected_utf8_bytes': int(dataset.record_byte_counts.sum())}
    (root / f'{stage}_data.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if stage == 'train':
        return train(model, optimizer, dataset, indices, config, progress)
    if stage == 'test':
        result = evaluate(model, dataset, config, indices, np.arange(len(dataset.records)), stage)
        (root / 'test_summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result
    raise ValueError(f'Unsupported stage: {stage}')


def train(model, optimizer, dataset, indices, config, progress):
    settings, options = config['training'], config['execution']
    root = Path(config['output']['root'])
    lengths = dataset.lengths(indices)
    total_steps = settings['epochs'] * math.ceil(len(indices) / settings['batch_size'])
    start = time.perf_counter()
    start_tokens = progress['tokens']
    with (root / 'metrics.jsonl').open('a') as journal:
        while progress['step'] < total_steps:
            plan = length_bucketed(indices, lengths, settings['batch_size'], config['seed'], progress['epoch'], 0)
            batches = remaining_batches(plan, progress['cursor'])
            loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collator(config['batching']),
                                num_workers=options['num_workers'], pin_memory=options['pin_memory'],
                                generator=torch.Generator().manual_seed(config['seed']))
            update_started = time.perf_counter()
            for batch in loader:
                model.train()
                batch = on_device(batch, options['device'])
                rate = warmup_cosine(progress['step'], total_steps, settings['learning_rate'],
                                     config['schedule']['warmup_fraction'], config['schedule']['end_factor'])
                optimizer.set_learning_rate(rate)
                optimizer.zero_grad(set_to_none=True)
                loss, values, valid, content = objective(model, batch, config)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step(options['device_type'])
                progress['step'] += 1
                count = int(valid.sum())
                samples = batch['input_ids'].shape[0]
                progress['cursor'] += samples
                progress['samples'] += samples
                progress['tokens'] += count
                progress['input_positions'] += batch['input_ids'].numel()
                progress['training_seconds'] += time.perf_counter() - update_started
                elapsed = time.perf_counter() - start
                record = {'step': progress['step'], 'epoch': progress['epoch'], 'loss': float(loss.detach()),
                          'content_nll': float((values.detach() * content).sum() / content.sum()),
                          'gradient_norm': float(norm), 'batch_samples': samples, 'samples': progress['samples'],
                          'target_tokens': count, 'input_positions': progress['input_positions'],
                          'wall_seconds': elapsed, 'training_seconds': progress['training_seconds'],
                          'tokens_per_second': (progress['tokens'] - start_tokens) / elapsed,
                          'learning_rate': rate, 'optimizer_seconds': optimizer.seconds(options['device_type'])}
                journal.write(json.dumps(record) + '\n')
                journal.flush()
                if progress['step'] % settings['log_every'] == 0:
                    print(json.dumps(record), flush=True)
                if progress['step'] % settings['checkpoint_interval'] == 0 and progress['step'] < total_steps:
                    save(model, optimizer, progress, config)
                update_started = time.perf_counter()
            progress['cursor'] = 0
            progress['epoch'] += 1
    save(model, optimizer, progress, config)
    result = {**progress, 'stop_reason': 'epochs_complete', 'processed_windows': progress['samples'],
              'sampled_data_passes': progress['samples'] / len(indices), 'wall_seconds': time.perf_counter() - start}
    (root / 'train_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result
