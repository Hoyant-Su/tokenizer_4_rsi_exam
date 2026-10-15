import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib import import_module
from multiprocessing import get_context

import pyarrow as pa


def initialize_encoder(load, settings, threads):
    global worker_tokenizer
    native_threads(threads)
    worker_tokenizer = load(settings)


def encode_text(text, add_special_tokens):
    return worker_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def process_text_encoder(load, tokenizer_settings, preparation, stack, add_special_tokens):
    pool = stack.enter_context(ProcessPoolExecutor(
        max_workers=preparation['cpu_workers'], mp_context=get_context(preparation['process_start']),
        initializer=initialize_encoder, initargs=(load, tokenizer_settings, preparation['worker_threads'])))
    encode = partial(encode_text, add_special_tokens=add_special_tokens)
    return lambda texts: list(pool.map(encode, texts, chunksize=preparation['chunksize']))


def encode_question(record, tokenizer, config):
    return (tokenizer.encode(record['input'] + config['corpus']['answer_prefix'], add_special_tokens=False).ids,
            tokenizer.encode(record['answer'], add_special_tokens=False).ids)


def batch_questions(tokenizer, records, prefixes):
    prompts = tokenizer.encode_batch(
        [row['input'] + prefix for row, prefix in zip(records, prefixes, strict=True)],
        add_special_tokens=False)
    answers = tokenizer.encode_batch([row['answer'] for row in records], add_special_tokens=False)
    return prompts, answers


def encode_record(record, tokenizer, config):
    return tokenizer.encode(record['text'], add_special_tokens=False)


def question_encoder(config, split):
    function = config['corpus']['question_encoders'][split]
    return getattr(import_module(function['module']), function['function'])


def native_threads(count):
    value = str(count)
    os.environ.update(TOKENIZERS_PARALLELISM='true', RAYON_NUM_THREADS=value,
                      OMP_NUM_THREADS=value, MKL_NUM_THREADS=value)
    pa.set_cpu_count(count)


def cpu_affinity(count):
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [available[index] for index in range(count)])


def cpu_budget(count):
    cpu_affinity(count)
    native_threads(count)
