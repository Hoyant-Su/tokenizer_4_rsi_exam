import json
import resource
from bisect import bisect_right
from contextlib import ExitStack
from itertools import islice
from importlib import import_module
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import orjson
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from data.language_model.batching import require_context, window_lengths
from data.language_model.encoding import cpu_budget, process_text_encoder
from data.streams.reading import Corpus
from model.language_model.tokenization import load as load_tokenizer


def deterministic_encoder(config, split, stack):
    tokenizer = load_tokenizer(config['tokenizer'])
    return lambda batch: tokenizer.encode_batch([row['text'] for _, _, row in batch], add_special_tokens=False)


def parallel_encoder(config, split, stack):
    cpu_budget(config['preparation']['cpu_workers'])
    return deterministic_encoder(config, split, stack)


def process_encoder(config, split, stack):
    settings = config['preparation']
    cpu_budget(settings['cpu_workers'])
    encode = process_text_encoder(load_tokenizer, config['tokenizer'], settings, stack, False)
    return lambda batch: [SimpleNamespace(ids=ids) for ids in encode(row['text'] for _, _, row in batch)]


def window_index(token_counts, context, dtype):
    lengths = np.asarray(token_counts, dtype=np.int64)
    longest = int(lengths.argmax())
    require_context(int(lengths[longest]), context, longest)
    counts = np.ones(len(lengths), dtype=np.int64)
    windows = np.zeros((len(lengths), 2), dtype=dtype)
    windows[:, 0] = np.arange(len(lengths))
    return windows, counts


def prepare_windows(config, split):
    started = perf_counter()
    settings = config['storage']
    paths = settings['paths_by_split'][split]
    source_path = settings['window_source_summaries_by_split'][split]
    source = json.loads(Path(source_path).read_text())
    table = np.memmap(paths['records'], mode='r', dtype=np.dtype([tuple(field) for field in settings['record_dtype']]))
    context = config['training']['context_length']
    windows, counts = window_index(table['token_count'], context, np.dtype(settings['window_dtype']))
    Path(paths['windows']).parent.mkdir(parents=True, exist_ok=True)
    windows.tofile(paths['windows'])
    source_counts = np.bincount(table['dataset'], weights=counts, minlength=len(source['datasets'])).astype(np.int64)
    result = {**source, 'context_length': context, 'windows': len(windows),
              'source_summary': {name: {**source['source_summary'][name], 'windows': int(source_counts[index])}
                                 for index, name in enumerate(source['datasets'])},
              'window_preparation': {'source_summary': source_path, 'wall_seconds': perf_counter() - started,
                                     'scope': 'Rebuilt windows only; unchanged token IDs, record metadata and identities.'}}
    Path(paths['summary']).parent.mkdir(parents=True, exist_ok=True)
    Path(paths['summary']).write_text(json.dumps(result, indent=settings['json_indent']) + '\n')
    print(json.dumps(result), flush=True)
    return result


def prepare(config, split):
    settings = config['storage']
    paths = settings['paths_by_split'][split]
    token_dtype = np.dtype(settings['token_dtype'])
    record_dtype = np.dtype([tuple(field) for field in settings['record_dtype']])
    window_dtype = np.dtype(settings['window_dtype'])
    context = config['training']['context_length']
    cpu_started = [resource.getrusage(scope) for scope in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    started = perf_counter()
    counts = {'records': 0, 'windows': 0, 'tokens': 0, 'content_tokens': 0, 'utf8_bytes': 0}
    datasets = {}
    source_summary = {}
    for path in paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        encoding = settings['encoder']
        encode = getattr(import_module(encoding['module']), encoding['function'])(config, split, stack)
        files = {name: stack.enter_context(Path(paths[name]).open('wb'))
                 for name in ('tokens', 'records', 'windows', 'identities')}

        def write(batch):
            encoded = encode(batch)
            metadata, windows, payload = [], [], []
            for (source_index, source_row, row), encoding in zip(batch, encoded, strict=True):
                content = encoding.ids
                token_ids = [config['tokenizer']['bos_id'], *content, config['tokenizer']['eos_id']]
                require_context(len(token_ids), context, row['id'])
                identity = orjson.dumps(row['id']) + b'\n'
                dataset = row['dataset']
                if dataset not in datasets:
                    datasets[dataset] = len(datasets)
                    source_summary[dataset] = {'records': 0, 'windows': 0, 'utf8_bytes': 0, 'content_tokens': 0}
                byte_count = len(row['text'].encode('utf-8'))
                metadata.append((counts['tokens'], len(token_ids), byte_count, datasets[dataset],
                                 source_index, source_row, files['identities'].tell(), len(identity)))
                files['identities'].write(identity)
                windows.append((counts['records'], 0))
                payload.append(np.asarray(token_ids, dtype=token_dtype))
                counts['records'] += 1
                counts['windows'] += 1
                counts['tokens'] += len(token_ids)
                counts['content_tokens'] += len(content)
                counts['utf8_bytes'] += byte_count
                for key, value in (('records', 1), ('windows', 1),
                                   ('utf8_bytes', byte_count), ('content_tokens', len(content))):
                    source_summary[dataset][key] += value
            np.concatenate(payload).tofile(files['tokens'])
            np.asarray(metadata, dtype=record_dtype).tofile(files['records'])
            np.asarray(windows, dtype=window_dtype).tofile(files['windows'])

        batch, byte_count = [], 0
        for source_index, path in enumerate(config['reading']['paths_by_split'][split]):
            reading = {**config['reading'], 'paths_by_split': {split: [path]}}
            for source_row, row in enumerate(Corpus(reading, split)):
                size = len(row['text'].encode('utf-8'))
                if batch and (len(batch) == settings['batch_rows'] or byte_count + size > settings['batch_bytes']):
                    write(batch)
                    batch, byte_count = [], 0
                batch.append((source_index, source_row, row))
                byte_count += size
        if batch:
            write(batch)
    cpu_finished = [resource.getrusage(scope) for scope in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    result = {**counts, 'split': split, 'context_length': context, 'datasets': list(datasets),
              'cpu_user_seconds': sum(end.ru_utime - begin.ru_utime for begin, end in zip(cpu_started, cpu_finished)),
              'cpu_system_seconds': sum(end.ru_stime - begin.ru_stime for begin, end in zip(cpu_started, cpu_finished)),
              'parent_peak_resident_bytes': cpu_finished[0].ru_maxrss * 1024,
              'maximum_child_peak_resident_bytes': cpu_finished[1].ru_maxrss * 1024,
              'preparation': config['preparation'],
              'source_summary': source_summary,
              'source_paths': config['reading']['paths_by_split'][split],
              'source_format': config['reading']['format'], 'wall_seconds': perf_counter() - started}
    Path(paths['summary']).write_text(json.dumps(result, indent=settings['json_indent']) + '\n')
    print(json.dumps(result), flush=True)
    return result


class Metadata:
    def __init__(self, table, identity_path, datasets):
        self.table = table
        self.path = identity_path
        self.identities = np.memmap(identity_path, mode='r', dtype=np.uint8)
        self.datasets = datasets

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table[index]
        start = int(row['identity_start'])
        identity = orjson.loads(self.identities[start:start + int(row['identity_bytes'])].tobytes())
        return {'id': identity, 'dataset': self.datasets[int(row['dataset'])],
                'text_utf8_bytes': int(row['utf8_bytes'])}


class StoredWindows(Dataset):
    text_field = 'text'
    byte_scope = 'All UTF-8 bytes of complete evaluated records; each content token scored once; specials excluded.'

    def __init__(self, config, split):
        self.config = config
        self.split = split
        settings = config['storage']
        paths = settings['paths_by_split'][split]
        self.summary = json.loads(Path(paths['summary']).read_text())
        self.context = config['training']['context_length']
        self.pad = config['tokenizer']['pad_id']
        self.tokens = np.memmap(paths['tokens'], mode='r', dtype=settings['token_dtype'])
        self.table = np.memmap(paths['records'], mode='r',
                               dtype=np.dtype([tuple(field) for field in settings['record_dtype']]))
        self.windows = np.memmap(paths['windows'], mode='r', dtype=settings['window_dtype']).reshape(-1, 2)
        self.records = Metadata(self.table, paths['identities'], self.summary['datasets'])
        longest = int(self.table['token_count'].argmax())
        require_context(int(self.table[longest]['token_count']), self.context, self.records[longest]['id'])
        if (np.any(self.windows[:, 1] != 0) or
                not np.array_equal(self.windows[:, 0], np.arange(len(self.table)))):
            raise ValueError('Stored windows split complete records; rebuild this corpus for complete-record training.')
        self.byte_count = self.summary['utf8_bytes']
        self.dataset_names = self.summary['datasets']
        self.source_summary = self.summary['source_summary']
        self.record_datasets = self.table['dataset']
        self.record_sources = self.table['source']
        self.source_names = self.summary['source_paths']
        self.record_byte_counts = self.table['utf8_bytes']
        self.record_window_offsets = np.arange(len(self.table) + 1)

    def __len__(self):
        return len(self.windows)

    def window_datasets(self, indices):
        return self.table['dataset'][self.windows[indices, 0]].astype(np.int64)

    def lengths(self, indices):
        return window_lengths(self.windows[indices], self.table['token_count'].astype(np.int64), self.context)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        windows = self.windows[indices].astype(np.int64)
        records = self.table[windows[:, 0]]
        starts = windows[:, 1, None]
        positions = np.arange(self.context)[None, :]
        counts = records['token_count'].astype(np.int64)[:, None]
        offsets = records['token_start'].astype(np.int64)[:, None]
        valid = positions + starts < counts - 1
        addresses = offsets + np.minimum(starts + positions, counts - 1)
        next_addresses = offsets + np.minimum(starts + positions + 1, counts - 1)
        inputs = np.where(valid, self.tokens[addresses], self.pad).astype(np.int64)
        targets = np.where(valid, self.tokens[next_addresses], self.pad).astype(np.int64)
        return {'input_ids': torch.from_numpy(inputs), 'target_ids': torch.from_numpy(targets),
                'attention_mask': torch.from_numpy(valid), 'target_mask': torch.from_numpy(valid.copy()),
                'content_mask': torch.from_numpy(valid & (positions + starts + 1 < counts - 1)),
                'window_index': torch.from_numpy(indices),
                'record_index': torch.from_numpy(windows[:, 0]),
                'dataset_index': torch.from_numpy(records['dataset'].astype(np.int64)),
                'source_index': torch.from_numpy(records['source'].astype(np.int64))}

    def __getitem__(self, index):
        return {name: value[0] for name, value in self.batch([index]).items()}

    def __getitems__(self, indices):
        batch = self.batch(indices)
        return [{name: value[index] for name, value in batch.items()} for index in range(len(indices))]

    def original(self, index):
        row = self.table[index]
        path = self.summary['source_paths'][int(row['source'])]
        ordinal = int(row['source_row'])
        if self.summary['source_format'] == 'jsonl':
            reading = {**self.config['reading'], 'paths_by_split': {self.split: [path]}}
            return next(islice(Corpus(reading, self.split), ordinal, ordinal + 1))
        with pq.ParquetFile(path) as file:
            ends = np.cumsum([file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)])
            group = bisect_right(ends, ordinal)
            start = 0 if group == 0 else int(ends[group - 1])
            return {**file.read_row_group(group).slice(ordinal - start, 1).to_pylist()[0], 'split': self.split}

    def generation_example(self, index, settings):
        record, start = map(int, self.windows[index])
        row = self.table[record]
        begin = int(row['token_start']) + start
        stop = min(begin + self.context + 1, int(row['token_start'] + row['token_count']))
        tokens = self.tokens[begin:stop].tolist()
        length = min(settings['prompt_tokens'], len(tokens) - 1)
        return self.original(record), tokens, tokens[:length]

    def generation_text(self, record, tokens, prompt, tokenizer):
        return {'prompt': tokenizer.decode(prompt, skip_special_tokens=True),
                'reference': tokenizer.decode(tokens[len(prompt):], skip_special_tokens=True),
                'reference_scope': 'Decoded tokenized window continuation.'}


def run(config, stage):
    operation = {'prepare_tokens': prepare, 'prepare_windows': prepare_windows}[stage]
    return [operation(config, split) for split in config['storage']['splits']]
