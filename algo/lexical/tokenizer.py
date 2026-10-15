import importlib
import json
import os
import random
import resource
from collections import defaultdict
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from time import perf_counter

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from data.streams.reading import Corpus
from data.language_model.encoding import native_threads, process_text_encoder

def pre_tokenizer(component):
    options = component['options'].copy()
    if component['name'] == 'Split':
        options['pattern'] = Regex(options['pattern'])
    return getattr(pre_tokenizers, component['name'])(**options)


def build(settings):
    model = {'bpe': models.BPE, 'unigram': models.Unigram}[settings['method']]
    trainer = {'bpe': trainers.BpeTrainer, 'unigram': trainers.UnigramTrainer}[settings['method']]
    tokenizer = Tokenizer(model(**settings['model']))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizer(component)
        for component in settings['pre_tokenizer_prefix']
    ] + [pre_tokenizers.ByteLevel(**settings['pre_tokenizer'])])
    tokenizer.decoder = decoders.ByteLevel()
    special = [AddedToken(value['text'], **settings['special_token_options'])
               for value in sorted(settings['special_tokens'], key=lambda token: token['id'])]
    fitting = trainer(vocab_size=settings['vocab_size'], show_progress=settings['show_progress'],
                      special_tokens=special, initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
                      **settings['trainer'])
    return tokenizer, fitting


def fit_tokens(rows, settings, seed):
    tokenizer, trainer = build(settings)
    tokenizer.train_from_iterator((row['text'] for row in rows), trainer=trainer, length=len(rows))
    return tokenizer, {}


def load_tokens(settings):
    return Tokenizer.from_file(settings['tokenizer_path'])


def test(config):
    native_threads(config['preparation']['cpu_workers'])
    settings = config['lexical']
    function = settings['load_function']
    load = getattr(importlib.import_module(function['module']), function['entry'])
    tokenizer = load(settings)
    rows = iter(Corpus(config['reading'], settings['test_split']))
    totals = defaultdict(lambda: {'records': 0, 'utf8_bytes': 0, 'characters': 0, 'tokens': 0,
                                 'roundtrip_failures': 0})
    encode_seconds, decode_seconds = 0.0, 0.0
    with ExitStack() as stack, Path(settings['records_path']).open('w', encoding='utf-8') as output:
        encode = {
            'direct': lambda: lambda texts: [item.ids for item in tokenizer.encode_batch(
                texts, add_special_tokens=settings['add_special_tokens'])],
            'process': lambda: process_text_encoder(load, settings, config['preparation'], stack,
                                                    settings['add_special_tokens']),
        }[settings['test_encoding']]()
        while batch := list(islice(rows, settings['batch_size'])):
            texts = [row['text'] for row in batch]
            started = perf_counter()
            ids = encode(texts)
            encode_seconds += perf_counter() - started
            started = perf_counter()
            decoded = tokenizer.decode_batch(ids, skip_special_tokens=settings['skip_special_tokens'])
            decode_seconds += perf_counter() - started
            for row, token_ids, restored in zip(batch, ids, decoded, strict=True):
                size = len(row['text'].encode('utf-8'))
                exact = restored.encode('utf-8') == row['text'].encode('utf-8')
                counts = totals[row['dataset']]
                counts['records'] += 1
                counts['utf8_bytes'] += size
                counts['characters'] += len(row['text'])
                counts['tokens'] += len(token_ids)
                counts['roundtrip_failures'] += int(not exact)
                record = {'id': row['id'], 'dataset': row['dataset'], 'split': row['split'],
                    'utf8_bytes': size, 'characters': len(row['text']), 'token_count': len(token_ids),
                    'roundtrip_exact': exact}
                if settings['save_token_ids']:
                    record['token_ids'] = token_ids
                output.write(json.dumps(record) + '\n')
    overall = {name: sum(value[name] for value in totals.values()) for name in next(iter(totals.values()))}
    result = {'method': settings['method'], 'split': settings['test_split'], 'datasets': dict(totals),
              'overall': overall, 'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
              'encode_utf8_bytes_per_second': overall['utf8_bytes'] / encode_seconds,
              'decode_utf8_bytes_per_second': overall['utf8_bytes'] / decode_seconds,
              'encode_tokens_per_second': overall['tokens'] / encode_seconds,
              'decode_tokens_per_second': overall['tokens'] / decode_seconds,
              'vocabulary_size': tokenizer.get_vocab_size(), 'environment': settings['environment']}
    result['peak_process_resident_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result['encoding_execution'] = {'mode': settings['test_encoding'], 'preparation': config['preparation']}
    Path(settings['test_summary_path']).write_text(json.dumps(result, indent=settings['json_indent']) + '\n')
    return result


def run(config, stage):
    settings = config['lexical']
    os.environ.update({name: str(value) for name, value in settings['environment'].items()})
    random.seed(config['seed'])
    Path(settings['output_root']).mkdir(parents=True, exist_ok=True)
    if stage == 'test':
        result = test(config)
    elif stage == 'fit':
        rows = Corpus(config['reading'], settings['train_split'])
        started = perf_counter()
        tokenizer, details = fit_tokens(rows, settings, config['seed'])
        tokenizer.save(settings['tokenizer_path'])
        result = {'method': settings['method'], 'seed': config['seed'],
                  'training_records': len(rows),
                  'training_utf8_bytes': sum(len(row['text'].encode('utf-8')) for row in rows),
                  'vocabulary_size': tokenizer.get_vocab_size(),
                  'fit_seconds': perf_counter() - started, 'method_details': details}
        Path(settings['fit_summary_path']).write_text(json.dumps(result, indent=settings['json_indent']) + '\n')
    else:
        raise ValueError(f'Unsupported tokenizer stage: {stage}')
    print(json.dumps(result), flush=True)
    return result
