import numpy as np
import torch

from model.language_model.alphabet import bytes_to_unicode
from model.language_model.tokenization import load as load_tokenizer


def literal_symbols(tokenizer, settings):
    alphabet = {character: byte for byte, character in bytes_to_unicode().items()}
    alphabet.update({symbol: len(bytes_to_unicode()) + index for index, symbol in enumerate(settings['structural_symbols'])})
    return {index: [alphabet[character] for character in piece] for piece, index in tokenizer.get_vocab().items()}


def symbol_rows(symbols, width, seed, standard_deviation):
    rows = np.empty((len(symbols), width), dtype=np.float64)
    for index, literal in symbols.items():
        entropy = np.random.SeedSequence([seed, len(literal), *literal])
        generator = np.random.Generator(np.random.PCG64(entropy))
        rows[index] = generator.normal(scale=standard_deviation, size=width)
    return rows


def initialize_literal_seeded(model, config):
    tokenizer = load_tokenizer(config['tokenizer'])
    settings = config['embedding_initialization']
    symbols = literal_symbols(tokenizer, settings)
    weight = model.get_input_embeddings().weight
    initialize_symbol_rows(weight, symbols, config['seed'], config['model']['initializer_range'],
                           config['tokenizer']['pad_id'])


def initialize_symbol_rows(weight, symbols, seed, standard_deviation, pad_id):
    indices = sorted(symbols)
    rows = symbol_rows({row: symbols[index] for row, index in enumerate(indices)},
                       weight.shape[1], seed, standard_deviation)
    with torch.no_grad():
        weight.index_copy_(0, torch.tensor(indices, device=weight.device),
                           torch.from_numpy(rows).to(device=weight.device, dtype=weight.dtype))
        weight[pad_id].zero_()
