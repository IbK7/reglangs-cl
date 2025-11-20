import os
import re
import sys
import zlib
import json
from collections import defaultdict

import torch
import random
import numpy as np
from tqdm import trange
from pathlib import Path
from itertools import groupby
import torch.nn.functional as nn
from torch.utils.data import random_split

class RecognitionDataset(torch.utils.data.Dataset):
    """
    Dataset for a language recognition problem given a random graph.
    """
    def __init__(self, inputs, outputs, lengths=None, out_lengths=None, alphabet_size=None, output_class_num=None):
        self.lengths = lengths
        self.out_lengths = out_lengths
        self.alphabet_size = alphabet_size
        self.output_class_num = output_class_num
        if self.lengths is not None:
            self.inputs = inputs
            self.outputs = outputs
        else:
            # onehot encoded tensors are dtype = torch.int64 but should be float or double
            self.inputs = torch.stack(inputs).float() if isinstance(inputs, tuple) else inputs
            self.outputs = torch.stack(outputs).float() if isinstance(outputs, tuple) else outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if is_packed, then we additionally return the length of the item
        if self.lengths is not None:
            if self.out_lengths is not None:
                return self.inputs[idx], self.outputs[idx], self.lengths[idx], self.out_lengths[idx], self.alphabet_size, self.output_class_num
            else:
                return self.inputs[idx], self.outputs[idx], self.lengths[idx], self.alphabet_size, self.output_class_num
        else:
            return self.inputs[idx], self.outputs[idx]

    def __getitems__(self, indices):
        # if is_packed, then we additionally return the length of the slice
        if self.lengths is not None:
            if self.out_lengths is not None:
                return self.inputs[indices], self.outputs[indices], self.lengths[indices], self.out_lengths[indices], self.alphabet_size, self.output_class_num
            else:
                return self.inputs[indices], self.outputs[indices], self.lengths[indices], self.alphabet_size, self.output_class_num
        else:
            return self.inputs[indices], self.outputs[indices]


def split_dataset(ins, outs):
    train_size = int(0.8 * len(ins))
    indices = np.arange(len(ins))
    train_indices = np.random.choice(indices, train_size, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    train_input, train_output = ins[train_indices], outs[train_indices]
    test_input, test_output = ins[test_indices], outs[test_indices]
    return train_input, train_output, test_input, test_output


def create_better_problems(automaton, max_length, max_train_length, train_batch_size, unseen_batch_size, save_dir, sampling="uniform-length", rng: int = 42, onehot=False):
    """
    max_length (int): the maximum length of the samples, including the unseen data split
    max_train_length (int): the maximum length of the training data
    batch_size (int): the number of samples per length
    save_dir (str): the directory to save the problems to
    rng (int): the random seed
    onehot (bool): whether to one-hot encode the inputs
    sampling (str): the sampling strategy to use; either "uniform-length" or "uniform" or "gaussian"
    """
    torch.manual_seed(rng)
    np.random.seed(rng)
    random.seed(rng)

    print(f"Creating problems for {save_dir} with {sampling} sampling...")
    if sampling == "uniform-length":
        train, unseen = uniform_length_sampling(automaton, max_length, max_train_length, train_batch_size,
                                                      unseen_batch_size, onehot)
    elif sampling == "uniform":
        train, unseen = uniform_sampling(automaton, max_length, max_train_length, train_batch_size,
                                               unseen_batch_size, onehot)
    elif sampling == "gaussian":
        train, test, unseen = gaussian_sampling(automaton, max_length, max_train_length, train_batch_size,
                                                unseen_batch_size, onehot)
    else:
        raise ValueError("Invalid sampling strategy, choose from 'uniform-length', 'uniform'.")

    train_input = torch.cat(train["train_input"], dim=0)
    train_output = torch.cat(train["train_output"], dim=0)
    train_output = train_output.unsqueeze(1)


    final_unseen_length_input = torch.cat(unseen["unseen_length_input"], dim=0)
    final_unseen_length_output = torch.cat(unseen["unseen_length_output"], dim=0)
    final_unseen_length_output = final_unseen_length_output.unsqueeze(1)

    if save_dir:
        save_problems(train_input, train_output, final_unseen_length_input, final_unseen_length_output, automaton,
                      str(save_dir), train["train_input_compression"],
                      unseen["unseen_length_input_compression"])
        print(f"Saved problems to {save_dir}.")
    else:
        return train_input, train_output, final_unseen_length_input, final_unseen_length_output


def gaussian_sampling(automaton, max_length, max_train_length, training_set_size, unseen_batch_size, onehot):
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    train_input_compression = []
    test_input_compression = []

    n_centers = 10
    std_dev = 0.5

    # Select n lengths from the possible range
    selected_lengths = np.random.choice(range(1, max_train_length + 1), size=n_centers, replace=False)
    print(f"Selected lengths for Gaussian sampling: {selected_lengths}")

    # Generate training samples based on Gaussian distribution
    samples_per_center = defaultdict(list)
    samples_per_length = defaultdict(list)
    for center in selected_lengths:
        mean = center
        num_samples = int(training_set_size / n_centers)

        while len(samples_per_length[center]) < num_samples:
            sample_lengths = np.random.normal(loc=mean, scale=std_dev, size=num_samples).round().astype(int)
            sample_lengths = sample_lengths[(sample_lengths > 0) & (sample_lengths <= max_train_length)]
            samples_per_length[center].extend(sample_lengths)
        samples_per_center[center] = samples_per_length[center][:num_samples]
        for entry in samples_per_center[center]:
            samples_per_length[entry].append(entry)


    # Create grouped lengths
    for length, samples in samples_per_length.items():
        print(f"Creating problems for length {length}...")
        unique_set = set()

        # Ensure uniqueness for shorter lengths
        if length <= 18:
            while len(unique_set) < len(samples):
                for entry in np.random.randint(0, len(automaton.sigma), (len(samples), max(samples)), dtype=np.int8):
                    unique_set.add(tuple(entry))
            np_rand = np.asarray(list(unique_set), dtype=np.int8)
        else:
            np_rand = np.random.randint(0, len(automaton.sigma), (len(samples), max(samples)), dtype=np.int8)

        # labels are binary for most problems, but not all
        labels = torch.tensor(automaton.get_output_labels_optimized(np_rand))
        outputs = nn.one_hot(labels.to(torch.int64), len(set(automaton.F))).type(torch.int8)
        strings = ["".join([automaton.sigma[i] for i in v]) for v in np_rand]
        compression_rates = [sys.getsizeof(s) / sys.getsizeof(zlib.compress(s.encode())) for s in strings]

        # Create one-hot encoding if specified
        if onehot:
            inputs = nn.one_hot(torch.Tensor(np_rand).long(), num_classes=len(automaton.sigma)).float()
            inputs = nn.pad(inputs, (0, 0, 0, max_train_length - inputs.size(1)))
        else:
            inputs = torch.from_numpy(np_rand).type(torch.int8)
            inputs = nn.pad(inputs, (0, max_train_length - inputs.shape[1]), value=-1)

        # Split into training and testing datasets
        if len(samples) > 1:
            no_test_samples = int(np.ceil(len(inputs) * 0.1))
            train_input.append(inputs[no_test_samples:])
            train_output.append(outputs[no_test_samples:])
            train_input_compression.extend(compression_rates[no_test_samples:])

            test_input.append(inputs[:no_test_samples])
            test_output.append(outputs[:no_test_samples])
            test_input_compression.extend(compression_rates[:no_test_samples])
        else:
            print(f"No samples of length {max(samples)} for test split")
            train_input.append(inputs)
            train_output.append(outputs)
            train_input_compression.extend(compression_rates)

    # Generate unseen lengths for testing
    unseen_length_input = []
    unseen_length_output = []
    unseen_length_input_compression = []
    for l in range(max_train_length + 1, max_length + 1):
        samples = np.random.randint(0, len(automaton.sigma), (unseen_batch_size, l), dtype=np.int8)
        # labels are binary for most problems, but not all
        labels = torch.tensor(automaton.get_output_labels_optimized(samples))
        outputs = nn.one_hot(labels.to(torch.int64), len(set(automaton.F))).type(torch.int8)
        unseen_length_output.append(outputs)
        strings = ["".join([automaton.sigma[i] for i in v]) for v in samples]
        unseen_length_input_compression.extend(
            [sys.getsizeof(s) / sys.getsizeof(zlib.compress(s.encode())) for s in strings])
        if onehot:
            onehot_input = nn.one_hot(torch.tensor(samples), num_classes=len(automaton.sigma)).type(
                torch.int8)
            inputs = nn.pad(onehot_input, (0, 0, 0, max_length - onehot_input.size(1)))
        else:
            samples = torch.from_numpy(samples).type(torch.int8)
            inputs = nn.pad(samples, (0, max_length - samples.shape[1]), value=-1)
        unseen_length_input.append(inputs)

    return (
        {"train_input": train_input, "train_output": train_output, "train_input_compression": train_input_compression},
        {"test_input": test_input, "test_output": test_output, "test_input_compression": test_input_compression},
        {"unseen_length_input": unseen_length_input, "unseen_length_output": unseen_length_output,
         "unseen_length_input_compression": unseen_length_input_compression}
    )


def uniform_sampling(automaton, max_length, max_train_length, training_set_size, unseen_batch_size, onehot):

    train_input = []
    train_output = []
    train_input_compression = []

    train_lengths = list(range(1, max_train_length + 1))
    total_vectors = sum(2 ** n for n in range(1, max_train_length + 1))
    probabilities = [(2 ** n) / total_vectors for n in range(1, max_train_length + 1)]
    train_selected_lengths = np.random.choice(train_lengths, size=training_set_size, p=probabilities)

    train_grouped_lengths = [list(group) for _, group in groupby(sorted(train_selected_lengths))]
    for samples in train_grouped_lengths:
        print(f"Creating problems for length {samples[0]}...")
        # ensure uniqueness for samples shorter than 18 symbols
        if samples[0] <= 18:
            unique_set = set()
            while len(unique_set) < len(samples):
                 for entry in np.random.randint(0, len(automaton.sigma), (len(samples), max(samples)), dtype=np.int8):
                        unique_set.add(tuple(entry))
            np_rand = np.asarray(list(unique_set), dtype=np.int8)
        else:
            # sample inputs
            np_rand = np.random.randint(0, len(automaton.sigma), (len(samples), max(samples)), dtype=np.int8)

        # labels are binary for most problems, but not all
        outputs = torch.tensor(automaton.get_output_labels_optimized(np_rand)).type(torch.int8)

        # calculate corresponding compression rates
        strings = ["".join([automaton.sigma[i] for i in v]) for v in np_rand]
        compression_rates = [sys.getsizeof(s) / sys.getsizeof(zlib.compress(s.encode())) for s in strings]

        # create one-hot encoding from the innermost dimension up to i and set the others to zeros
        if onehot:
            inputs = nn.one_hot(torch.Tensor(np_rand).long(), num_classes=len(automaton.sigma)).float()
            inputs = nn.pad(inputs, (0, 0, 0, max_train_length - inputs.size(1)))
        else:
            inputs = torch.from_numpy(np_rand).type(torch.int8)
            inputs = nn.pad(inputs, (0, max_train_length - inputs.shape[1]), value=-1)


        train_input.append(inputs)
        train_output.append(outputs)
        train_input_compression.extend(compression_rates)

    unseen_length_input = []
    unseen_length_output = []
    unseen_length_input_compression = []
    for l in range(max_train_length + 1, max_length + 1):
        print(f"Creating problems for length {l}...")
        samples = np.random.randint(0, len(automaton.sigma), (unseen_batch_size, l), dtype=np.int8)
        # labels are binary for most problems, but not all
        outputs = torch.tensor(automaton.get_output_labels_optimized(samples)).type(torch.int8)
        unseen_length_output.append(outputs)
        strings = ["".join([automaton.sigma[i] for i in v]) for v in samples]
        unseen_length_input_compression.extend([sys.getsizeof(s) / sys.getsizeof(zlib.compress(s.encode())) for s in strings])
        if onehot:
            onehot_input = nn.one_hot(torch.tensor(samples), num_classes=len(automaton.sigma)).type(torch.int8)
            inputs = nn.pad(onehot_input, (0, 0, 0, max_length - onehot_input.size(1)))
        else:
            samples = torch.from_numpy(samples).type(torch.int8)
            inputs = nn.pad(samples, (0, max_length - samples.shape[1]), value=-1)
        unseen_length_input.append(inputs)

    return (
    {"train_input": train_input, "train_output": train_output, "train_input_compression": train_input_compression},
    {"unseen_length_input": unseen_length_input, "unseen_length_output": unseen_length_output,
     "unseen_length_input_compression": unseen_length_input_compression})


def uniform_length_sampling(automaton, max_length, max_train_length, training_set_size, unseen_batch_size, onehot):
    train_input = []
    train_output = []
    train_input_compression = []
    unseen_length_input = []
    unseen_length_output = []
    unseen_length_input_compression = []

    # Calculate initial samples per length based on training set size and automaton properties
    max_samples_per_length = training_set_size // max_train_length
    temp_samples_per_length = [
        min(max_samples_per_length, len(automaton.sigma) ** length)
        for length in range(1, max_train_length + 1)
    ]

    # Identify lengths with fewer samples than the maximum allowable per length
    lengths_with_insufficient_samples = [
        samples for samples in temp_samples_per_length
        if samples < max_samples_per_length
    ]

    # Calculate remaining samples and allocate them to lengths with sufficient capacity
    remaining_samples = training_set_size - sum(lengths_with_insufficient_samples)
    sufficient_lengths_count = max_train_length - len(lengths_with_insufficient_samples)
    new_batch_size = remaining_samples // sufficient_lengths_count
    extra_samples = remaining_samples % sufficient_lengths_count

    # Finalize the distribution of samples per length
    samples_per_length = np.array(temp_samples_per_length)
    samples_per_length[samples_per_length >= max_samples_per_length] = new_batch_size
    samples_per_length[-extra_samples:] += 1

    # Verify the total number of allocated samples matches the training set size
    assert sum(samples_per_length) == training_set_size

    # Calculate test batch sizes and concatenate with training samples per length
    unseen_samples_per_length = np.full(max_length - max_train_length, unseen_batch_size)
    all_samples_per_length = np.concatenate((samples_per_length, unseen_samples_per_length))

    for l, batch_size in zip(range(1, max_length+1), all_samples_per_length):
        print(f"Creating problems for length {l}...")
        random_subset = set()  # Store unique vectors

        if batch_size > 30000 and l > 18:
            random_subset = np.random.randint(0, len(automaton.sigma),
                                              (min(batch_size, len(automaton.sigma) ** l), l))
        else:
            # Keep generating random vectors until reaching the desired subset size
            while len(random_subset) < min(batch_size, len(automaton.sigma) ** l):
                for row in np.random.randint(0, len(automaton.sigma),
                                             (min(batch_size, len(automaton.sigma) ** l), l)):
                    random_subset.add(tuple(row))

        # random_subset has shape (batch_size, l)
        random_subset = np.asarray(list(random_subset), dtype=np.int8)
        if len(random_subset) > batch_size:
            random_subset = random_subset[:batch_size]
        strings = ["".join([automaton.sigma[i] for i in v]) for v in random_subset]
        # get the compression rates for all the strings
        compression_rates = [sys.getsizeof(s) / sys.getsizeof(zlib.compress(s.encode())) for s in strings]

        # labels are binary for most problems, but not all
        outputs = torch.tensor(automaton.get_output_labels_optimized(random_subset)).type(torch.int8)
        # get argmax of ouputs
        outputs = torch.argmax(outputs, dim=1).type(torch.int8)

        padding_length = max_train_length if l <= max_train_length else max_length
        if onehot:
            onehot_input = nn.one_hot(torch.tensor(random_subset), num_classes=len(automaton.sigma)).type(torch.int8)
            inputs = nn.pad(onehot_input, (0, 0, 0, padding_length - onehot_input.size(1)))
        else:
            random_subset = torch.from_numpy(random_subset).type(torch.int8)
            inputs = nn.pad(random_subset, (0, padding_length - random_subset.shape[1]), value=-1)

        if l <= max_train_length:
            train_input.append(inputs)
            train_output.append(outputs)
            train_input_compression += compression_rates
        else:
            unseen_length_input.append(inputs)
            unseen_length_output.append(outputs)
            unseen_length_input_compression += compression_rates

    return ({"train_input": train_input, "train_output": train_output, "train_input_compression": train_input_compression},
            {"unseen_length_input": unseen_length_input, "unseen_length_output": unseen_length_output, "unseen_length_input_compression": unseen_length_input_compression})


def create_problems(language, max_length, batch_size, save_dir, complete_sampling=False, better_sampling=False, rng: int = 42):
    """
    Function to create problems for a given language. Implements three different sampling strategies:
    - naive (complete=False, better_sampling=False): returns always batch_size many samples
    - complete (complete=True): returns at least batch_size many samples (repetitions), but at max sigma^i many samples,
    potentially oversampling for small i
    - better_sampling (better_sampling=True): returns at max batch_size many samples, but at least sigma^i many samples
    leading to fewer repetitions; though will require buckets over lengths for training [1, 2, 3], [4, 5, 6], etc.

    ASSUMPTION: the automaton is fully connected, i.e. one outgoing edge for each state for each symbol in the alphabet.

    Args:
        language: the language for which to create the problems
        max_length: the maximum length of the samples
        batch_size: the number of samples per length
        save_dir: the directory to save the problems to
        complete_sampling: if True, returns at least batch_size many samples, but at max sigma^i many samples
        better_sampling: if True, returns at max batch_size many samples, but at least sigma^i many samples
        rng: the random seed

    Either saves the problem to disk or returns the input and output tensors.
    """
    dataset = []
    # returns at max batch_size many samples
    if better_sampling is True:
        for i in trange(1, max_length + 1):
            batch = language.sample_batch(rng, min(batch_size, len(language.automaton.sigma) ** i), i)
            dataset.append(batch)
    # returns at least complete many samples, but at max sigma^i many samples
    elif complete_sampling is True:
        for i in trange(1, max_length + 1):
            batch = language.sample_batch(rng, max(batch_size, len(language.automaton.sigma) ** i), i)
            dataset.append(batch)
    # returns always batch_size many samples
    elif complete_sampling is False and better_sampling is False:
        for i in trange(1, max_length + 1):
            batch = language.sample_batch(rng, batch_size, i)
            dataset.append(batch)
    else:
        raise ValueError("Invalid combination of parameters.")


    final_input, final_output = dataset_postprocessing(dataset)
    train_input, train_output, test_input, test_output = split_dataset(final_input, final_output)

    print(f"Created {len(train_input)} train and {len(test_input)} test problems.")

    if save_dir:
        save_problems(train_input, train_output, test_input, test_output, language, save_dir)
        print(f"Saved problems to {save_dir}.")
    else:
        return train_input, train_output, test_input, test_output


def dataset_postprocessing(dataset):
    # Find the maximum sequence length
    max_seq_length = max([d["input"].size(1) for d in dataset])

    # Pad the sequences in each batch to the maximum sequence length
    input_tensors = [torch.nn.functional.pad(d["input"], (0, 0, 0, max_seq_length - d["input"].size(1))) for d in
                     dataset]
    output_tensors = [d["output"] for d in dataset]

    final_input = torch.cat(input_tensors, dim=0)
    final_output = torch.cat(output_tensors, dim=0)
    return final_input, final_output


def save_problems(train_ins, train_outs, unseen_ins,
                  unseen_outs, automat, save_dir, train_input_compression=None,
                  unseen_input_compression=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(train_ins.type(torch.int8), os.path.join(save_dir, 'train_input.pt'))
    torch.save(train_outs.type(torch.int8), os.path.join(save_dir, 'train_output.pt'))
    torch.save(unseen_ins.type(torch.int8), os.path.join(save_dir, 'unseen_input.pt'))
    torch.save(unseen_outs.type(torch.int8), os.path.join(save_dir, 'unseen_output.pt'))
    np.save(os.path.join(save_dir, 'train_input_compression.npy'), np.asarray(train_input_compression))
    np.save(os.path.join(save_dir, 'unseen_input_compression.npy'), np.asarray(unseen_input_compression))

    automat.name = save_dir
    # Save the regular language object for reference
    torch.save(automat, os.path.join(save_dir, 'language.pt'))


def load_compression_data(save_dir) -> tuple[np.ndarray, np.ndarray]:
    train_input_compression = None
    unseen_input_compression = None
    if os.path.exists(os.path.join(save_dir, 'train_input_compression.npy')):
        train_input_compression = np.load(os.path.join(save_dir, 'train_input_compression.npy'))
    if os.path.exists(os.path.join(save_dir, 'unseen_input_compression.npy')):
        unseen_input_compression = np.load(os.path.join(save_dir, 'unseen_input_compression.npy'))
    return train_input_compression, unseen_input_compression


def load_language(save_dir):
    save_dir = Path(save_dir)
    # Regex pattern to match _2_5_2_Recognition_
    pattern = r'_(\d+)_(\d+)_(\d+)_Recognition_'

    # Replacement format
    def replacer(match):
        return f"_{match.group(1)}{match.group(2)}{match.group(3)}-Recognition_"

    # Replace in the path
    new_parts = [re.sub(pattern, replacer, part) for part in save_dir.parts]
    new_path = Path(*new_parts)

    return torch.load(os.path.join(new_path, 'language.pt'))


def load_state2idx_dicts(save_dir):
    with open(os.path.join(save_dir, 'train_states2idx.json'), 'r') as f:
        states2idx = json.load(f)
    return states2idx

def load_single_problem(save_dir, unseen=False, language_only=False):
    # Load the regular language object
    language = None
    if os.path.exists(os.path.join(save_dir, 'language.pt')):
        language = torch.load(os.path.join(save_dir, 'language.pt'), weights_only=False)

    if language_only:
        return language

    # Load the input and output tensors
    train_ins = torch.load(os.path.join(save_dir, 'train_input.pt'), weights_only=True)
    train_outs = torch.load(os.path.join(save_dir, 'train_output.pt'), weights_only=True)

    if unseen:
        unseen_ins = torch.load(os.path.join(save_dir, 'unseen_input.pt'))
        unseen_outs = torch.load(os.path.join(save_dir, 'unseen_output.pt'))
        return unseen_ins, unseen_outs
    else:
        return train_ins, train_outs, language


def filtered_load_problems(base_dir="/home/mlai32/toborek/datasets/RegularLanguages/", num_symbols=None,
                           num_states=None, max_length=None, num_final_states=None, seed=None):
    """
    If called without any arguments, loads all the datasets in the data directory.
    """
    datasets = []

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        # Extract parameters from the directory name
        dir_params = dir_name.split("_")
        dir_num_symbols = int(dir_params[0])
        dir_num_states = int(dir_params[1])
        dir_max_length = int(dir_params[2])
        dir_num_final_states = int(dir_params[3])
        dir_seed = int(dir_params[4])

        # Check if the directory matches the provided parameters
        if ((num_symbols is None or dir_num_symbols == num_symbols) and
                (num_states is None or dir_num_states == num_states) and
                (max_length is None or dir_max_length == max_length) and
                (num_final_states is None or dir_num_final_states == num_final_states) and
                (seed is None or dir_seed == seed)):
            # Load the input and output tensors
            train_ins, train_outs, test_ins, test_outs, language = load_single_problem(dir_path)

            datasets.append((train_ins, train_outs, test_ins, test_outs, language))

    return datasets
