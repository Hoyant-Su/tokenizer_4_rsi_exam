from functools import partial

import numpy as np
from torch.utils.data import default_collate


def require_context(length, context_length, identity):
    if length > context_length:
        raise ValueError(
            f'Complete sequence {identity!r} has {length} tokens including special tokens; '
            f'context limit is {context_length}. No truncation or record splitting is permitted.')


def window_lengths(windows, record_token_counts, context_length):
    return np.minimum(context_length, record_token_counts[windows[:, 0]] - windows[:, 1] - 1)


def length_bucketed(indices, lengths, batch_size, seed, epoch, block):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, block]))
    order = rng.permutation(indices)
    order = order[np.argsort(lengths[order], kind='stable')]
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    return [batches[index] for index in rng.permutation(len(batches))]


def remaining_batches(batches, consumed_windows):
    boundaries = np.r_[0, np.cumsum([len(batch) for batch in batches])].tolist()
    return batches[boundaries.index(consumed_windows):]


def collate_windows(examples, sequence_fields):
    batch = default_collate(examples)
    length = int(batch['attention_mask'].sum(dim=1).max())
    for name in sequence_fields:
        batch[name] = batch[name][:, :length]
    return batch


def collator(settings):
    return {'contiguous': lambda: default_collate,
            'length_bucketed': lambda: partial(collate_windows, sequence_fields=settings['sequence_fields'])}[settings['mode']]()
