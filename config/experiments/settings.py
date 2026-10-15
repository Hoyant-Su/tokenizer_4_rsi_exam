from pathlib import Path

import yaml


def merge(shared, declared):
    result = dict(shared)
    for name, value in declared.items():
        result[name] = merge(shared[name], value) if isinstance(value, dict) and name in shared and isinstance(shared[name], dict) else value
    return result


def load_config(path):
    path = Path(path)
    declared = yaml.safe_load(path.read_text())
    shared = declared.pop('shared_config')
    if shared is None:
        return declared
    parents = [shared] if isinstance(shared, str) else shared
    inherited = {}
    for parent in parents:
        inherited = merge(inherited, load_config(path.parent / parent))
    return merge(inherited, declared)
