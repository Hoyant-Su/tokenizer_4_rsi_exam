import torch
import torch.nn.functional as F
import transformers

from model.language_model.initialization import initialize_literal_seeded


def build(config, vocabulary_size):
    architecture = config['architecture']
    settings = getattr(transformers, architecture['config_class'])(
        vocab_size=vocabulary_size, **config['model'],
        pad_token_id=config['tokenizer']['pad_id'],
        bos_token_id=config['tokenizer']['bos_id'],
        eos_token_id=config['tokenizer']['eos_id'])
    model = getattr(transformers, architecture['model_class'])(settings).to(config['execution']['device'])
    initialize_literal_seeded(model, config)
    return model


def losses(model, batch, config):
    with torch.autocast(device_type=config['execution']['device_type'],
                        dtype=getattr(torch, config['execution']['dtype']),
                        enabled=config['execution']['autocast']):
        logits = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'], use_cache=False).logits
    token_loss = token_losses(logits, batch)
    return token_loss, batch['target_mask'], batch['content_mask']


def token_losses(logits, batch):
    values = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch['target_ids'].reshape(-1),
                            reduction='none').reshape_as(batch['target_ids'])
    torch._assert_async(torch.isfinite(values).all(), 'Non-finite token likelihood; computation cannot produce valid losses or scores.')
    return values


def objective(model, batch, config):
    values, _, content = losses(model, batch, config)
    valid = batch[config['training']['loss_mask']]
    return (values * valid).sum() / valid.sum(), values, valid, content
