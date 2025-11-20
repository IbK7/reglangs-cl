import torch
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from utils.util import TorchRandomSeed
from itertools import chain, combinations

from problems.problem import RecognitionDataset, load_single_problem, load_compression_data


def check_format(sample):
    """
    Determine if the sample is in the old (one-hot, float64, zero-padding)
    or new (binary, int8, -1 padding) format.
    """
    if sample.dtype == torch.int8 and sample.ndim < 2:
        # Likely the new format (binary vectors, int8)
        # not checking for padding as there might be samples without padding
        return 'new_format'
    elif sample.ndim == 2: # torch.is_floating_point(sample) and
        # Likely the old format (one-hot encoding, float64)
        return 'old_format'
    # If neither, raise an error or return a default
    raise ValueError("Unrecognized data format.")


def count_non_padding(sample):
    """
    Count the number of non-padding elements in a sample.
    """
    format_type = check_format(sample)
    if format_type == 'old_format':
        return (sample != torch.zeros_like(sample[0, :])).any(dim=-1).sum().item()
    elif format_type == 'new_format':
        return (sample != -1).sum().item()


def count_non_padding_batch(batch) -> torch.Tensor:
    """
    Count the number of non-padding elements in a batch. Returns a tensor containing the length for each sample in the
    batch, i.e. of shape (batch_size,).
    """
    format_type = check_format(batch[0])
    if format_type == 'old_format':
        return (batch != torch.zeros_like(batch[0, 0, :])).any(dim=-1).sum(dim=1)
    elif format_type == 'new_format':
        return (batch != -1).sum(dim=1)


def select_samples_of_max_length(dataset, length):
    selected_samples = []
    for sample in dataset:
        # sample[0] is the input, sample[1] is the output
        if count_non_padding(sample[0]) <= length:
            selected_samples.append(sample)
    return RecognitionDataset(*zip(*selected_samples))


def select_samples_of_min_length(dataset, length):
    selected_samples = []
    for sample in dataset:
        if count_non_padding(sample[0]) >= length:
            selected_samples.append(sample)
    return RecognitionDataset(*zip(*selected_samples))


def select_samples_of_length(dataset, length):
    selected_samples = []
    for sample in dataset:
        if count_non_padding(sample[0]) == length:
            selected_samples.append(sample)
    return RecognitionDataset(*zip(*selected_samples))


def get_length2index_dict(dataset) -> dict:
    """
    Get a dictionary of the form sample_length: [indices] for a given dataset. Expects padding to be 0 (old format)
    or -1 (new format).
    """
    length_tensor = count_non_padding_batch(dataset)
    length_tensor = length_tensor.numpy()
    length_to_idx = {value: np.where(length_tensor == value)[0] for value in np.unique(length_tensor)}
    return length_to_idx


def get_state2index_dict(train_ins, language, allowed_indices=None) -> dict:
    """
    Get a dictionary of the form state_tuple: [indices] for a given dataset.
    """
    states_to_idx = defaultdict(list)
    states = language.numeric_to_states_batch(train_ins).cpu().numpy()
    print("States created")
    if allowed_indices is not None:
        filtered_indices = [ix for ix in range(len(states)) if ix in allowed_indices]
        filtered_states = states[filtered_indices]
    else:
        filtered_indices = range(len(states))
        filtered_states = states
    print("Filtered indices created")

    for idx_pos, ix in tqdm(enumerate(filtered_indices), total=len(filtered_indices)):
        # Convert the state to a frozenset to use it as a key in the dictionary
        state = filtered_states[idx_pos]
        key = tuple(int(i) for i in set(state[state != -1]))
        states_to_idx[key].append(ix)
    # frozenset_states = [frozenset(state[state != -1].tolist()) for state in filtered_states]
    # print("Frozenset states created")
    # for ix, state in tqdm(zip(filtered_indices, frozenset_states)):
    #     states_to_idx[state].append(ix)
    return states_to_idx


def old_load_training_data(params) -> tuple[RecognitionDataset, RecognitionDataset]:
    """
    The returned datasets are Subsets of the original dataset, with the indices shuffled
    """
    if params.get("train_data", None) is None:
        # Load problem dataset
        # Define the dataset
        train_ins, train_outs, test_ins, test_outs, language = load_single_problem(params["problem"])
        # Split dataset into training and validation
        train_size = int(params["train_val_split"] * len(train_ins))
        indices = np.arange(len(train_ins))
        np.random.shuffle(indices)
        # Split the dataset into training and validation sets
        train_index, val_index = indices[:train_size], indices[train_size:]
        orig_train_dataset = RecognitionDataset(train_ins[train_index], train_outs[train_index])
        orig_val_dataset = RecognitionDataset(train_ins[val_index], train_outs[val_index])
    else:
        orig_train_dataset = params["train_data"]
        orig_val_dataset = params["val_data"]

    # in case we want to train on only a fraction of the training data; validation data is not subsampled
    if params.get("training_subsampling", None):
        train_indices = orig_train_dataset.indices
        orig_train_dataset = orig_train_dataset[torch.tensor(train_indices[:int(len(train_indices) * params["training_subsampling"])])]

    if params.get("subset_size", None) and params.get("subset_seed", None) is not None:
        with TorchRandomSeed(params["subset_seed"]):
            np.random.seed(params["subset_seed"])
            orig_train_dataset = orig_train_dataset[torch.tensor(
                np.random.choice(list(range(len(orig_train_dataset))),
                                 int(len(orig_train_dataset) * params["subset_size"]),
                                 replace=False))]

    return orig_train_dataset, orig_val_dataset


def load_training_data(params) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The returned datasets are Subsets of the original dataset, with the indices shuffled
    """
    if params.get("train_data", None) is None:
        # Load problem dataset
        # Define the dataset
        train_ins, train_outs, val_ins, val_outs, test_ins, test_outs, language = load_single_problem(params["problem"])
    # check should be done outside this function in the future
    else:
        # params["*data"] should contain raw tensors instead of RecognitionDataset objects
        train_ins, train_outs = params["train_data"]
        val_ins, val_outs = params["val_data"]

    # in case we want to train on only a fraction of the training data; validation data is not subsampled
    if params.get("training_subsampling", None):
        train_indices = list(range(len(train_ins)))
        train_ins, train_outs = (train_ins[torch.tensor(train_indices[:int(len(train_indices) * params["training_subsampling"])])],
                                 train_outs[torch.tensor(train_indices[:int(len(train_indices) * params["training_subsampling"])])])

    if params.get("subset_size", None) and params.get("subset_seed", None) is not None:
        with TorchRandomSeed(params["subset_seed"]):
            np.random.seed(params["subset_seed"])
            train_ins = train_ins[torch.tensor(np.random.choice(list(range(len(train_ins))),
                                 int(len(train_ins) * params["subset_size"]),
                                 replace=False))]

    return train_ins, train_outs, val_ins, val_outs


def load_language(path):
    # loads an automaton, RandomAutomaton or FiniteAutomaton
    _, _, _, _, _, _, language = load_single_problem(path)
    return language


def load_test_data(params) -> RecognitionDataset:
    _, _, _, _, test_ins, test_outs, _ = load_single_problem(params["problem"])
    return RecognitionDataset(test_ins, test_outs)


def load_used_train_data(run_path) -> RecognitionDataset:
    """
    Load the original training dataset used for training the model.
    """
    return torch.load(run_path / "orig_train_dataset.pt")


def get_outputs_from_subset(subset):
    return subset.dataset.dataset.outputs


def calculate_random_baseline_from_subset(subset):
    return subset.outputs.float().mean(dim=0).max().item()


def get_curriculum_buckets(distributed, train_ins, train_outs, train_len2idx, exclusive=False, return_lengths=True):
    """
    Create training and validation buckets based on distributed lengths over number of buckets. Exclusive means that the
    buckets are exclusive, i.e. the indices are not shared between buckets.
    """
    train_buckets, train_idx, train_lengths = [], [], []
    #sorting the samples into buckets with differently long input; first bucket will contain samples up to length 10, second
    #up to 20, final bucket contains the entire dataset
    num_indices_per_bucket = {i : 0 for i in range(len(distributed))}
    for i, sizes in enumerate(distributed):
        if exclusive:
            for size in sizes:
                num_indices_per_bucket[i] += train_len2idx[size].size
        else:
            for size in sizes:
                for key, value in num_indices_per_bucket.items():
                    if key >= i:
                        num_indices_per_bucket[key] += train_len2idx[size].size

    train_buckets = []
    # create numpy arrays in the train buckets with correct sizes

    for i, sizes in enumerate(distributed):
        if return_lengths:
            train_buckets.append((torch.zeros(num_indices_per_bucket[i], train_ins.size(1)),
                                  torch.zeros(num_indices_per_bucket[i], train_outs.size(1)),
                                  torch.zeros(num_indices_per_bucket[i], dtype=torch.int64),
                                  torch.zeros(num_indices_per_bucket[i], dtype=torch.int64)))
        else:
            train_buckets.append((torch.zeros(num_indices_per_bucket[i], train_ins.size(1)),
                                  torch.zeros(num_indices_per_bucket[i], train_outs.size(1)),
                                  None,
                                  None))

    # get the indices and fill the buckets
    current_start_index = 0
    for i, sizes in enumerate(distributed):
        if exclusive:
            current_start_index = 0
            for size in sizes:
                train_buckets[i][0][current_start_index:current_start_index + train_len2idx[size].size] = train_ins[train_len2idx[size]]
                train_buckets[i][1][current_start_index:current_start_index + train_len2idx[size].size] = train_outs[train_len2idx[size]]
                if return_lengths:
                    train_buckets[i][2][current_start_index:current_start_index + train_len2idx[size].size] = torch.tensor([size] * train_len2idx[size].size, dtype=torch.int64)
                    train_buckets[i][3][current_start_index:current_start_index + train_len2idx[size].size] = count_non_padding_batch(train_outs[train_len2idx[size]]).to(torch.int64)
                current_start_index += train_len2idx[size].size
        else:
            for size in sizes:
                for key, value in num_indices_per_bucket.items():
                    if key >= i:
                        train_buckets[key][0][current_start_index:current_start_index + train_len2idx[size].size] = train_ins[train_len2idx[size]]
                        train_buckets[key][1][current_start_index:current_start_index + train_len2idx[size].size] = train_outs[train_len2idx[size]]
                        if return_lengths:
                            train_buckets[key][2][current_start_index:current_start_index + train_len2idx[size].size] = torch.tensor([size] * train_len2idx[size].size, dtype=torch.int64)
                            train_buckets[key][3][current_start_index:current_start_index + train_len2idx[size].size] = count_non_padding_batch(train_outs[train_len2idx[size]]).to(torch.int64)
                        current_start_index += train_len2idx[size].size
    return train_buckets



    # #sorting the samples into buckets with differently long input; first bucket will contain samples up to length 10, second
    # #up to 20, final bucket contains the entire dataset
    # train_buckets, train_idx, train_lengths = [], [], []
    # for sizes in distributed:
    #     if exclusive:
    #         train_idx, train_lengths = [], []
    #     # iterate over all required lengths for the current bucket
    #     for s in sizes:
    #         # defaultdict, will return empty list if there is no entry for the given length
    #         relevant_indices = train_len2idx[s]
    #         train_idx.extend(relevant_indices)
    #         train_lengths.extend(torch.tensor([s] * len(relevant_indices)))
    #     # each bucket's training data is a subset of the original dataset
    #     # each bucket consists of a tuple of (train_ins, train_outs, lengths)
    #     # without the list() call, the old Subset object be updated with the new indices
    #     out_lengths = count_non_padding_batch(train_outs[train_idx])
    #     if return_lengths:
    #         train_buckets.append((train_ins[train_idx], train_outs[train_idx], torch.tensor(train_lengths), out_lengths))
    #     else:
    #         train_buckets.append((train_ins[train_idx], train_outs[train_idx], None, None))
    # return train_buckets


def get_no_curr_buckets(train_ins, train_outs, return_lengths=True):
    if return_lengths:
        train_lengths = count_non_padding_batch(train_ins)
        out_lengths = count_non_padding_batch(train_outs)
        return [(train_ins, train_outs, train_lengths, out_lengths)]
    else:
        return [(train_ins, train_outs, None, None)]

def get_simple_dfa_buckets(dfa, train_ins, train_outs, train_states2idx):
    train_buckets = []
    sorted_state_sets = sorted(train_states2idx.keys(), key=lambda x: (len(x), x))

    for state_set in sorted_state_sets:
        # check if the current state set is a subset of the added states
        indices = train_states2idx[state_set]
        train_lengths = count_non_padding_batch(train_ins[indices])
        out_lengths = count_non_padding_batch(train_outs[indices])
        train_buckets.append((train_ins[indices], train_outs[indices], train_lengths, out_lengths))
    return train_buckets

def get_dfa_buckets(dfa, train_ins, train_outs, train_states2idx, exclusive=False, return_lengths=True):
    """
    Create training and validation buckets based on dfa state combinations. Exclusive means that the buckets are
    exclusive, i.e. the indices are not shared between buckets.
    """
    train_buckets, train_idx = [], []
    added_states = set()
    # iterate over the buckets defined by the dfa.json file
    for bucket in dfa:
        if exclusive:
            train_idx = []
        # add current bucket's states to the set of all states
        added_states = added_states.union(set(bucket))
        # get all possible combinations of the added states
        all_state_combinations = list(chain(*[combinations(added_states, r) for r in range(1, len(added_states) + 1)]))
        # filter such that combinations include at least one state present in the current bucket
        filtered_state_combinations = [s for s in all_state_combinations if any(value in bucket for value in s)]

        for s in filtered_state_combinations:
            indices = train_states2idx.get(tuple(s), [])
            if indices:
                train_idx.extend(indices)

        # check if there are any indices, if not continue
        if not train_idx:
            continue

        if return_lengths:
            train_lengths = count_non_padding_batch(train_ins[train_idx])
            out_lengths = count_non_padding_batch(train_outs[train_idx])
            train_buckets.append((train_ins[train_idx], train_outs[train_idx], train_lengths, out_lengths))
        else:
            train_buckets.append((train_ins[train_idx], train_outs[train_idx], None))
    return train_buckets


def get_compression_buckets(train_ins, train_outs, data, params, exclusive: bool = True, return_lengths: bool = True):
    """
    Create training and validation buckets based on the compression rates of the samples.
    """
    # get the training and validation compression rates
    train_compression2idx = defaultdict(list)
    train_compression_set = set()
    for i, idx in enumerate(train_ins):
        # round to six decimal places
        t_comp = round(data['train_compression'][i], 6)
        train_compression_set.add(t_comp)
        train_compression2idx[t_comp].append(i)

    # sort the compression rates
    reversed_sort = False
    if 'anti' in params["training_strategy"]:
        reversed_sort = True
    train_compression_set = sorted(list(train_compression_set), reverse=reversed_sort)

    index_list = []
    for comp in train_compression_set:
        index_list.extend(train_compression2idx[comp])

    indices = np.array(index_list)
    bucket_number = params["num_buckets"]
    train_bucket_idx = np.array_split(indices, bucket_number)
    train_buckets = []
    train_idx = []
    for i in range(params["num_buckets"]):
        if exclusive:
            train_idx = []
        train_idx.extend(train_bucket_idx[i])
        if return_lengths:
            train_lengths = count_non_padding_batch(train_ins[train_idx])
            train_buckets.append((train_ins[train_idx], train_outs[train_idx], train_lengths))
        else:
            train_buckets.append((train_ins[train_idx], train_outs[train_idx], None))
    return train_buckets


def get_random_buckets(train_ins, train_outs, params, exclusive: bool = True, return_lengths: bool = True):
    """
    Create training and validation buckets based on random indices.
    """
    # set the random seed using params["seed"]
    np.random.seed(params["seed"])

    # split train_idx and val_idx into bucket_number parts
    indices = np.random.permutation(len(train_ins))
    bucket_number = params["num_buckets"]
    train_bucket_idx = np.array_split(indices, bucket_number)
    train_buckets, train_idx = [], []
    for i in range(params["num_buckets"]):
        if exclusive:
            train_idx = []
        train_idx.extend(train_bucket_idx[i])
        if return_lengths:
            train_lengths = count_non_padding_batch(train_ins[train_idx])
            train_buckets.append((train_ins[train_idx], train_outs[train_idx], train_lengths))
        else:
            train_buckets.append((train_ins[train_idx], train_outs[train_idx], None))
    return train_buckets


def prepare_data(bucket_ins, bucket_outs, bucket_lengths, updates_per_bucket, params, strategy=None):
    """
    Oversamples the data if there are too few samples, selects a subset if there are too many
    """

    #total_samples_needed = updates_per_bucket * params["training_params"]["batch_size"]
    #if len(bucket_ins) < total_samples_needed:
    #    # If we need more samples than available, repeat the data
    #    indices = torch.randint(0, len(bucket_ins), (total_samples_needed,))
    #    data_to_use = (bucket_ins[indices], bucket_outs[indices], bucket_lengths[indices])
    #else:
    #    data_to_use = (bucket_ins, bucket_outs, bucket_lengths)
    if strategy == 'uniform-length':
        unique_lengths = torch.unique(bucket_lengths)
        unique_length_indices = [torch.where(bucket_lengths == length)[0] for length in unique_lengths]
        max_length = max([len(v) for v in unique_length_indices])
        # repeat the data such that each length has the same number of samples, i.e., max_length
        indices = None
        for i, unique_length in enumerate(unique_length_indices):
            if len(unique_length) < max_length:
                if i == 0:
                    indices = np.random.choice(unique_length, max_length, replace=True)
                else:
                    indices = np.concatenate((indices, np.random.choice(unique_length, max_length, replace=True)))
            else:
                if i == 0:
                    indices = np.array(unique_length_indices[i])
                else:
                    indices = np.concatenate((indices, np.array(unique_length_indices[i])))
        data_to_use = (bucket_ins[indices], bucket_outs[indices], bucket_lengths[indices])
    else:
        data_to_use = (bucket_ins, bucket_outs, bucket_lengths)

    return data_to_use
