import json
import random
from pathlib import Path

import numpy as np
import torch


def save(model, optimizer, progress, config):
    root = Path(config['output']['checkpoint_root'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"step_{progress['step']:09d}.pt"
    temporary = path.with_suffix('.partial')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'progress': progress, 'config': config, 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if config['execution']['device_type'] == 'cuda' else []}, temporary)
    temporary.replace(path)
    index = {'path': str(path), **progress}
    index_path = Path(config['output']['root']) / 'checkpoint.json'
    index_temporary = index_path.with_suffix('.partial')
    index_temporary.write_text(json.dumps(index, indent=2) + '\n')
    index_temporary.replace(index_path)
    # Reserve one checkpoint slot for the next write; committed state remains readable during it.
    committed = sorted(root.glob('step_*.pt'), reverse=True)
    for obsolete in committed[config['training']['max_checkpoint_num'] - 1:]:
        obsolete.unlink()


def restore(model, optimizer, path, config):
    state = torch.load(path, map_location=config['execution']['device'], weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if config['execution']['device_type'] == 'cuda':
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
    return state['progress']
