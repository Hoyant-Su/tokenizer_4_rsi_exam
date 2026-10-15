import math

import torch

from algo.optimizers.moonlight import Muon as NativeMuon, zeropower_via_newtonschulz5


class Muon(NativeMuon):
    def __init__(self, lr, wd, muon_params, momentum, nesterov, ns_steps,
                 adamw_params, adamw_betas, adamw_eps):
        super().__init__(lr=lr, wd=wd, muon_params=muon_params, momentum=momentum,
                         nesterov=nesterov, ns_steps=ns_steps, adamw_params=adamw_params,
                         adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        self.graphs = {}

    def orthogonalize(self, direction, steps):
        key = (direction.device, direction.dtype, direction.shape, steps)
        if key not in self.graphs:
            inputs = direction.clone()
            outputs = [zeropower_via_newtonschulz5(matrix, steps) for matrix in inputs.unbind()]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = torch.stack([zeropower_via_newtonschulz5(matrix, steps) for matrix in inputs.unbind()])
            self.graphs[key] = (inputs, outputs, graph)
        inputs, outputs, graph = self.graphs[key]
        inputs.copy_(direction)
        graph.replay()
        return outputs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = {}
            for parameter in group['params']:
                if not self.state[parameter]['use_muon']:
                    raise ValueError('Grouped Muon requires AdamW parameters in the separate shared optimizer.')
                if parameter.grad is not None:
                    buckets.setdefault((parameter.device, parameter.dtype, parameter.shape), []).append(parameter)
            for parameters in buckets.values():
                gradients = [parameter.grad for parameter in parameters]
                for parameter in parameters:
                    if 'momentum_buffer' not in self.state[parameter]:
                        self.state[parameter]['momentum_buffer'] = torch.zeros_like(parameter)
                buffers = [self.state[parameter]['momentum_buffer'] for parameter in parameters]
                torch._foreach_mul_(buffers, group['momentum'])
                torch._foreach_add_(buffers, gradients)
                direction = (torch._foreach_add(gradients, buffers, alpha=group['momentum'])
                             if group['nesterov'] else buffers)
                updates = self.orthogonalize(torch.stack(direction), group['ns_steps']).unbind()
                rate = group['lr'] * (0.2 * math.sqrt(max(parameters[0].shape)))
                torch._foreach_mul_(parameters, 1 - group['lr'] * group['wd'])
                torch._foreach_add_(parameters, updates, alpha=-rate)
        return loss
