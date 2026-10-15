def on_device(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}
