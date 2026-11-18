import torch

def make_batch(batch):
    inputs, targets = zip(*batch)

    # Longueurs réelles des trajectoires
    lengths = torch.tensor(
        [x.shape[1] for x in inputs],
        dtype=torch.long
    )

    # Padding dynamique
    max_length = int(lengths.max())

    padded_inputs = torch.zeros(
        len(inputs), 2, max_length,
        dtype=torch.float32
    )

    for i, x in enumerate(inputs):
        padded_inputs[i, :, :x.shape[1]] = x

    targets = torch.stack(targets)

    return padded_inputs, targets, lengths