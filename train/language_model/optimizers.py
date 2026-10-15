from fnmatch import fnmatchcase
from time import perf_counter

import torch

from algo.optimizers.cuda_graph import Muon


class Optimizers:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def set_learning_rate(self, rate):
        for group in self.param_groups:
            group['lr'] = rate * group['lr_mult']

    def step(self, device_type):
        self.timings = {}
        for name, optimizer in self.optimizers.items():
            if device_type == 'cuda':
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                optimizer.step()
                end.record()
                self.timings[name] = (start, end)
            else:
                start = perf_counter()
                optimizer.step()
                self.timings[name] = perf_counter() - start

    def seconds(self, device_type):
        if device_type == 'cuda':
            return {name: start.elapsed_time(end) / 1000 for name, (start, end) in self.timings.items()}
        return self.timings

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state):
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])


def parameter_groups(model, patterns):
    linear_types = (torch.nn.Linear,)
    protected = {id(model.get_input_embeddings().weight), id(model.get_output_embeddings().weight)}
    hidden = {id(layer.weight) for name, module in model.named_modules()
              if any(fnmatchcase(name, pattern) for pattern in patterns)
              for layer in module.modules() if isinstance(layer, linear_types)} - protected
    parameters = list(model.parameters())
    return ([parameter for parameter in parameters if id(parameter) in hidden],
            [parameter for parameter in parameters if id(parameter) not in hidden])


def build(model, config):
    settings, options = config['training'], config['optimizer']

    def adamw(parameters):
        groups = [{'params': list(parameters), 'lr_mult': options['parameter_learning_rate_multiplier'], 'lr': settings['learning_rate']}]
        return torch.optim.AdamW(groups, lr=settings['learning_rate'], betas=tuple(settings['betas']),
                                 eps=settings['eps'], weight_decay=settings['weight_decay'],
                                 fused=settings['fused_optimizer'])

    implementation = options['muon']
    hidden, other = parameter_groups(model, implementation['module_boundaries'])
    muon = Muon(lr=settings['learning_rate'], wd=settings['weight_decay'], muon_params=hidden,
                       adamw_params=[], momentum=implementation['momentum'], nesterov=implementation['nesterov'],
                       ns_steps=implementation['ns_steps'], adamw_betas=tuple(settings['betas']),
                       adamw_eps=settings['eps'])
    muon.param_groups[0]['lr_mult'] = options['parameter_learning_rate_multiplier']
    return Optimizers({'muon': muon, 'adamw': adamw(other)})
