import math


def warmup_cosine(completed_updates, total_updates, peak, warmup_fraction, end_factor):
    update = completed_updates + 1
    warmup = math.ceil(total_updates * warmup_fraction)
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (total_updates - warmup)
    return peak * (end_factor + (1 - end_factor) * (1 + math.cos(math.pi * progress)) / 2)
