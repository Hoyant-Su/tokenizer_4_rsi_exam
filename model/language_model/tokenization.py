from tokenizers import Tokenizer

from config.experiments.settings import load_config


def load(settings):
    lexical = load_config(settings['lexical_config'])['lexical']
    tokenizer = Tokenizer.from_file(lexical['tokenizer_path'])
    tokenizer.decode_errors = settings['decode_errors']
    return tokenizer
