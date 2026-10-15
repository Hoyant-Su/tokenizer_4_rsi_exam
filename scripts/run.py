import argparse
import importlib
import os

from config.experiments.settings import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if 'execution' in config:
        os.environ.update(config['execution']['environment'])
    workflow = config['workflow']
    module = importlib.import_module(workflow['module'])
    getattr(module, workflow['entry'])(config, arguments.stage)


if __name__ == '__main__':
    main()
