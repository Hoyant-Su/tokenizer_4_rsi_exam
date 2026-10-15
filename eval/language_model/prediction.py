import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.language_model.batching import collator
from data.language_model.corpus import on_device
from model.language_model.causal import losses


@torch.inference_mode()
def evaluate(model, dataset, config, window_indices, record_indices, stage):
    model.eval()
    progress = config['evaluation']['progress']
    records = config['evaluation']['records']
    save_records = {'disabled': False, 'jsonl': True}[records['mode'][stage]]
    persistent = {'disabled': False, 'fresh': True, 'resume': True}[progress['mode']]
    cursor = 0
    elapsed = 0.0
    groupings = config['evaluation']['groupings']
    names = {group: getattr(dataset, group + '_names') for group in groupings}
    totals = {group: torch.zeros((len(names[group]), 4), dtype=torch.float64,
                                device=config['execution']['device']) for group in groupings}
    record_totals = torch.zeros((len(dataset.records), 4), dtype=torch.float64,
                               device=config['execution']['device']) if save_records else None
    if persistent:
        path = Path(config['output']['root']) / progress['state_file']
        path.parent.mkdir(parents=True, exist_ok=True)
        if progress['mode'] == 'resume':
            state = json.loads(path.read_text())
            cursor, elapsed = state['next_window'], state['elapsed_seconds']
            totals = {group: torch.tensor(state['totals'][group], dtype=torch.float64,
                                         device=config['execution']['device']) for group in groupings}
            if save_records:
                record_totals = torch.tensor(state['record_totals'], dtype=torch.float64,
                                             device=config['execution']['device'])
    loader = DataLoader(Subset(dataset, window_indices[cursor:]), batch_size=config['evaluation']['batch_size'], shuffle=False,
                        collate_fn=collator(config['batching']),
                        num_workers=config['execution']['num_workers'],
                        pin_memory=config['execution']['pin_memory'],
                        generator=torch.Generator().manual_seed(config['seed']))
    started = perf_counter()
    batches = math.ceil(cursor / config['evaluation']['batch_size'])
    for batch in loader:
        batch = on_device(batch, config['execution']['device'])
        values, valid, content = losses(model, batch, config)
        sample_totals = torch.stack(((values * valid).sum(dim=1, dtype=torch.float64), valid.sum(dim=1),
                                    (values * content).sum(dim=1, dtype=torch.float64), content.sum(dim=1)), dim=1)
        for group in groupings:
            totals[group].index_add_(0, batch[group + '_index'], sample_totals)
        if save_records:
            record_totals.index_add_(0, batch['record_index'], sample_totals)
        cursor += len(batch['input_ids'])
        batches += 1
        if persistent and (batches % progress['interval_batches'] == 0 or cursor == len(window_indices)):
            state = {'next_window': cursor, 'next_batch': batches, 'total_windows': len(window_indices),
                     'elapsed_seconds': elapsed + perf_counter() - started,
                     'totals': {group: value.tolist() for group, value in totals.items()}}
            if save_records:
                state['record_totals'] = record_totals.tolist()
            temporary = path.with_suffix('.partial')
            temporary.write_text(json.dumps(state) + '\n')
            temporary.replace(path)
    full_nll, full_tokens, content_nll, content_tokens = totals[groupings[0]].sum(dim=0).tolist()
    complete = record_indices is not None
    byte_total = int(dataset.record_byte_counts[record_indices].sum()) if complete else None
    result = {'records': len(record_indices) if complete else len(dataset.records), 'windows': len(window_indices),
              'nll_including_eos': full_nll / full_tokens if full_tokens else None,
              'content_nll': content_nll / content_tokens if content_tokens else None, 'content_tokens': int(content_tokens),
              'original_utf8_bytes': byte_total,
              'bits_per_original_utf8_byte': content_nll / math.log(2) / byte_total if byte_total else None,
              'byte_denominator_scope': (dataset.byte_scope if complete else
                                        'Not reported: selected overfit windows do not cover complete records.')}
    for group in groupings:
        owners = getattr(dataset, 'record_' + group + 's')
        selected = record_indices if complete else np.arange(len(dataset.records))
        record_counts = np.bincount(owners[selected], minlength=len(names[group]))
        byte_counts = np.zeros(len(names[group]), dtype=np.int64)
        np.add.at(byte_counts, owners[selected], dataset.record_byte_counts[selected].astype(np.int64))
        rows = {}
        for index, (name, (nll, count, content_sum, content_count)) in enumerate(zip(names[group], totals[group].tolist(), strict=True)):
            rows[name] = {'source_records': int(record_counts[index]), 'target_tokens': int(count),
                          'content_tokens': int(content_count), 'nll_including_eos': nll / count if count else None,
                          'content_nll': content_sum / content_count if content_count else None,
                          'original_utf8_bytes': int(byte_counts[index]) if complete else None,
                          'bits_per_original_utf8_byte': (content_sum / math.log(2) / int(byte_counts[index])
                                                         if complete and byte_counts[index] else None)}
        result[group + 's'] = rows
    if save_records:
        selected = record_indices if complete else np.unique(np.asarray(dataset.windows)[window_indices, 0])
        values = record_totals.cpu().numpy()
        output = Path(config['output']['root']) / records['file']
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.partial')
        with temporary.open('w') as stream:
            for index in selected:
                record = dataset.records[int(index)]
                nll, count, content_sum, content_count = values[index]
                byte_count = int(dataset.record_byte_counts[index])
                row = {'id': record['id'], 'dataset': record['dataset'], 'record_index': int(index),
                       'source': dataset.source_names[dataset.record_sources[index]],
                       'target_nll_sum': float(nll), 'target_tokens': int(count),
                       'content_nll_sum': float(content_sum), 'content_tokens': int(content_count),
                       'original_utf8_bytes': byte_count, 'complete_record': complete,
                       'bits_per_original_utf8_byte': (content_sum / math.log(2) / byte_count
                                                      if complete and byte_count else None)}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        temporary.replace(output)
        result['record_metrics'] = {'path': str(output), 'records': len(selected),
                                    'byte_denominator_scope': result['byte_denominator_scope']}
    return result
