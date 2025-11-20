import torch
import random
import numpy as np
from pathlib import Path
import torch.nn.utils.rnn as rnn_utils

from problems.problem import RecognitionDataset


class TorchRandomSeed(object):
    """
    Thank you, Sebastian :)
    Class to be used when opening a with clause. On enter sets the random seed for torch based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = torch.get_rng_state()
        torch.set_rng_state(torch.manual_seed(self.seed).get_state())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_rng_state(self.prev_random_state)


def set_random_seeds(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_batches(dataset: RecognitionDataset, batch_size, shuffle=True) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create batches from the dataset
    """
    # TODO fix DataLoader problems
    #return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    if shuffle:
        indices = np.random.permutation(len(dataset))
    else:
        indices = np.arange(len(dataset))
    # create batches
    if len(dataset) < batch_size:
        return [(dataset.inputs[indices], dataset.outputs[indices])]
    indices = np.array_split(indices, len(dataset) // batch_size)
    return dataset.inputs[indices], dataset.outputs[indices]


def collate_batch(batch):
    return batch[0], batch[1], None


def collate_packed(batch):
    sequences, labels, lengths = batch
    packed_sequences = rnn_utils.pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
    # packed_labels = rnn_utils.pack_padded_sequence(labels.unsqueeze(1), lengths, batch_first=True, enforce_sorted=False)
    return packed_sequences, labels, lengths


def collate_packed(batch):
    """
    Collate function that packs the sequences, but handles the new data format, i.e. does onehot encoding and
    transforms the padding values from -1 to the zero vector
    """
    sequences, labels, lengths, out_lengths, alphabet_size, num_output_classes = batch
    label_length = labels.size(1)
    # padd sequences with label_length -1 at the end
    sequences = torch.nn.functional.pad(sequences, (0, label_length), value=-1)
    # transform inputs to onehot encoded, packed, float tensors
    padding_indices = sequences == -1

    sequences = torch.where(-1 == sequences, torch.tensor(0), sequences).long()
    sequences = torch.nn.functional.one_hot(sequences, num_classes=alphabet_size).float()
    sequences[padding_indices] = 1
    packed_sequences = rnn_utils.pack_padded_sequence(sequences, lengths+out_lengths, batch_first=True, enforce_sorted=False)
    label_padding_indices = labels == -1
    labels[label_padding_indices] = 0
    # turn labels to long
    labels = labels.long()
    # one hot encode labels
    labels = torch.nn.functional.one_hot(labels, num_classes=num_output_classes).float()
    labels[label_padding_indices] = 0
    # turns labels into float
    labels = labels.float()

    return packed_sequences, labels, lengths, out_lengths, alphabet_size, num_output_classes


def collate_transformer(batch):
    """
    Collate function for Transformer models. Pads sequences and labels to uniform length,
    converts to one-hot, and creates padding masks.
    """
    sequences, labels, lengths, out_lengths, alphabet_size, num_output_classes = batch
    device = sequences.device

    # Determine total sequence length (input + output)
    total_lengths = lengths + out_lengths
    max_len = total_lengths.max().item()

    # Pad input sequences with -1 to max_len
    sequences = torch.nn.functional.pad(sequences, (0, max_len - sequences.size(1)), value=-1)

    # Replace -1 with 0 for one-hot encoding, and generate one-hot vectors
    padding_mask = sequences == -1  # (batch_size, max_len)
    sequences = torch.where(sequences == -1, torch.tensor(0, device=device), sequences)
    sequences = torch.nn.functional.one_hot(sequences.to(torch.int64), num_classes=alphabet_size).float()
    sequences[padding_mask] = 1  # all-ones "neutral" token

    # Pad labels similarly to match max_len (or label-specific max if desired)
    label_length = labels.size(1)
    labels = torch.nn.functional.pad(labels, (0, max_len - label_length), value=-1)
    label_padding_mask = labels == -1
    labels = torch.where(label_padding_mask, torch.tensor(0, device=device), labels).long()
    labels = torch.nn.functional.one_hot(labels, num_classes=num_output_classes).float()
    labels[label_padding_mask] = 0

    return sequences, labels, lengths, out_lengths, alphabet_size, num_output_classes


# Function to get a unique file path by appending a counter
def get_unique_path(directory: Path, filename):
    counter = 1
    new_filename = Path(filename)
    new_filepath = directory / new_filename
    while new_filepath.exists():
        new_filename = f"{filename.stem}_{counter}{filename.suffix}"
        new_filepath = directory / new_filename
        counter += 1
    return new_filepath
