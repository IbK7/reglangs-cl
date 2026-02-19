## train_models.py
import os
import sys
import yaml
import copy
import time
import json
import torch
import pickle
import random
import logging
import hashlib
import warnings
import numpy as np
from pathlib import Path
from itertools import chain
from tqdm import trange, tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.lstm import LSTM
from models.rnn import RNN
from models.transformer_relative import Transformer
from models.bert import BERT
from problems.problem import RecognitionDataset
from models.training import evaluate, set_training_paths, reset_to_checkpoint, train_one_greedy_batch, collect_results
from utils.dataset import get_length2index_dict, select_samples_of_max_length, select_samples_of_min_length, \
    load_training_data, get_curriculum_buckets, calculate_random_baseline_from_subset, \
    load_single_problem, get_simple_dfa_buckets, get_no_curr_buckets, count_non_padding_batch
from utils.util import TorchRandomSeed, collate_packed

implemented_strategies = ["no-curr", "curr", "anti", "single", "uniform"]
# verbosity to logging levels
verbosity_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
myrange = {0: range, 1: trange, 2: trange}
# Dictionary to select the model
select_model = {"RNN": RNN, "LSTM": LSTM, "TransformerEncoder": Transformer, "TransformerRelative": Transformer, "BERT": BERT}
select_collate = {'RNN': collate_packed, 'LSTM': collate_packed, 'TransformerEncoder': collate_packed, "TransformerRelative": collate_packed, 'BERT': collate_packed} # 'CNN': collate_packed,
selection_function = {1: select_samples_of_max_length, -1: select_samples_of_min_length, 0: select_samples_of_max_length}


def hash_model_parameters(model):
    model_state_dict = model.state_dict()
    model_parameters_string = str(model_state_dict)
    model_hash = hashlib.md5(model_parameters_string.encode()).hexdigest()
    return model_hash


def set_random_seeds(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(base_path, data, params, model_params, config, device):
    # Configure logging
    logging.basicConfig(level=verbosity_levels[params["verbosity"]], format='%(message)s')
    # Path for the experiment
    experiment_path, best_model_path, params_path, train_dataset_path, val_dataset_path = set_training_paths(
        base_path, params)
    # Set random seeds for reproducibility
    set_random_seeds(params["seed"])

    # LOAD DATASETS
    train_ins, train_outs = data["train_data"]
    val_ins, val_outs = data["val_data"]
    language = load_single_problem(params["problem"], language_only=True)
    val_lengths = count_non_padding_batch(val_ins)
    val_out_lengths = count_non_padding_batch(val_outs)

    # for problems that don't come with an automaton
    if language is not None:
        output_class_num = len(set(language.F))
        alphabet_size = len(set(language.sigma))
    else:
        unique_columns = train_ins[:, 0].unique()
        output_class_num = len(unique_columns)
        alphabet_size = len(unique_columns)

    # PREPARE VALIDATION DATA SPLIT
    val_dataset = RecognitionDataset(val_ins, val_outs, val_lengths, val_out_lengths, alphabet_size=alphabet_size, output_class_num=output_class_num)
    # still need the following lines even if config['metric'] is 'state', because I might want to limit the input length
    # and still do a curriculum with state as metric
    train_len2idx, val_len2idx = get_length2index_dict(train_ins), get_length2index_dict(val_ins)

    # FILTER TRAINING DATA IF NECESSARY
    max_input_length = params.get('max_input_length', 40)
    if max_input_length < 40:
        train_len2idx = {k: v for k, v in train_len2idx.items() if k <= max_input_length}
        idxs_of_length = list(train_len2idx.values())
        train_ins = train_ins[np.concatenate(idxs_of_length)]

    if config['metric'] == 'length':
        total_units = len(train_len2idx)
        relevant_dict = train_len2idx
    elif config['metric'] == 'states':
        # load dfa.json for the respective problem under the problem's path
        dfa = json.load(open(params["problem"] + "/dfa.json", "r"))
        with open(params["problem"] + "/train_states2idx.pkl", "rb") as f:
            train_states2idx = pickle.load(f)

        # improves random sampling later on
        for k, v in train_states2idx.items():
            train_states2idx[k] = np.asarray(v)

        total_units = len(train_states2idx)
        relevant_dict = train_states2idx
    else:
        raise ValueError(f"Invalid metric: {config['metric']}, should be one of ['length', 'states'].")

    # samples per unit (either length or states): needed for uniform sampling
    samples_per_unit = model_params[params["model"]]["batch_size"] / total_units

    # PREPARE UNSEEN DATA SPLIT
    unseen_dataset = data.get("unseen_data", None)
    unseen_dataset.out_lengths = count_non_padding_batch(unseen_dataset.outputs)
    unseen_dataset.alphabet_size = val_dataset.alphabet_size
    unseen_dataset.output_class_num = val_dataset.output_class_num

    if max_input_length < 20 and config['metric'] == 'length':
        params['num_buckets'] = max_input_length

    train_buckets = None
    # PREPARE BUCKETS
    # CURRICULUM STRATEGIES
    if config['metric'] == 'length' and params['training_strategy'] not in ["no-curr", "uniform", "single"]:
        if params["training_strategy"] in ["curr", "anti"]:
            # creates a list of arrays with increasing lengths, each array contains the lengths that go into one bucket
            # if max_length is 40 and num_progressions is 5, the lengths will be distributed as follows:
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ... , [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
            min_length = min(train_len2idx.keys())
            max_length = max(train_len2idx.keys())
            distributed = np.array_split(np.arange(min_length, max_length + 1), params["num_buckets"])
            ## ANTI CURRICULUM
            if params["training_strategy"] == "anti":
                distributed = list(reversed(distributed))
            # train_buckets contains tuples of (ins, outs, lengths)
            train_buckets = get_curriculum_buckets(distributed, train_ins, train_outs, train_len2idx,
                                                   exclusive=params["exclusive"],
                                                   return_lengths=params.get("pack_sequences", False))
    elif config['metric'] == 'states' and params['training_strategy'] not in ["no-curr", "uniform", "single"]:
        if params["training_strategy"] in ["curr", "anti"]:
            train_buckets = get_simple_dfa_buckets(dfa,
                                                    train_ins,
                                                    train_outs,
                                                    train_states2idx,
                                                    )
            params['num_buckets'] = len(train_buckets)
            if params["training_strategy"] == "anti":
                train_buckets = list(reversed(train_buckets))

    # NO CURRICULUM STRATEGIES either with length or states as metric
    elif params["training_strategy"] in ["no-curr", "uniform", "single"]:
        # for train, we use the entire dataset up to max length in one bucket doesn't matter which metric we use later on
        train_buckets = get_no_curr_buckets(train_ins, train_outs, return_lengths=params.get("pack_sequences", False))
    else:
        raise ValueError(f"Invalid training strategy: {params['training_strategy']}, should be one of {implemented_strategies}.")

    # MODEL INITIALISATION
    with TorchRandomSeed(params.get("model_seed", params["seed"])):
        if params["model"] == "RNN":
            model = RNN(**{**model_params[params["model"]], **params["model_params"]})
            model.flatten_parameters()
        elif params["model"] == "LSTM":
            model = LSTM(**{**model_params[params["model"]], **params["model_params"]})
            model.flatten_parameters()
        elif params["model"] == "TransformerEncoder":
            model = Transformer(**{**model_params[params["model"]], **params["model_params"]})
        elif params["model"] == "TransformerRelative":
            model = Transformer(**{**model_params[params["model"]], **params["model_params"]})
        elif params["model"] == "BERT":
            model = BERT(**{**model_params[params["model"]], **params["model_params"]})
        model.to(device)

    # Loss function and optimizer
    if model_params[params["model"]]["loss_fn"] == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif model_params[params["model"]]["loss_fn"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Invalid loss function: {params['model_params'][params['model']]['loss_fn']}, should be one of ['cross_entropy', 'mse'].")

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params[params["model"]]["lr"])

    scheduler = None
    if model_params[params["model"]]['scheduler']:
        if model_params[params["model"]]['scheduler'] in ['ReduceLROnPlateau', 'reduce_lr_on_plateau']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        elif model_params[params["model"]]['scheduler'] in ['StepLR', 'step_lr']:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["eval_every_n_batches"], gamma=0.5)
        elif model_params[params["model"]]['scheduler'] in ['Linear', 'LinearLR']:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.999, end_factor=0.001, total_iters=params["num_updates"])
        else:
            print("No scheduler is used.")


    # Select batch size wrt the model architecture
    eval_batch_size = len(val_dataset) if params["eval_batch_size"] == -1 else min(params["eval_batch_size"], len(val_dataset))
    if params.get("model") == "CNN":
        eval_batch_size = len(val_dataset)
    elif params.get("model") in ["TransformerEncoder", "TransformerRelative", "BERT"]:
        eval_batch_size = 8

    total_batches = 0

    # Tensorboard writer
    writer = SummaryWriter(log_dir=str(experiment_path))
    logging.debug(params)

    # Initialize metrics dictionary
    results = defaultdict(list)

    # parametrised for curr strategies except dfa and no-curr
    number_of_buckets = len(train_buckets)
    # array of arrays with the number of updates per bucket
    base = params["num_updates"] // number_of_buckets
    remainder = params["num_updates"] % number_of_buckets
    updates_per_bucket = [base + 1 if i < remainder else base for i in range(number_of_buckets)]
    logging.debug(f"Starting training with {number_of_buckets} buckets and {updates_per_bucket} updates per bucket.")

    # PREPARE VALIDATION DATA
    val_loader = DataLoader(val_dataset,
                            batch_size=eval_batch_size,
                            collate_fn=select_collate[params['model']],
                            shuffle=False,
                            num_workers=0)

    # PREPARE UNSEEN DATA
    unseen_loader = DataLoader(unseen_dataset,
                               batch_size=eval_batch_size,
                               collate_fn=select_collate[params['model']],
                               shuffle=False,
                               num_workers=0)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    for bucket_id in range(number_of_buckets):
        start_time = time.time()

        if params["continue_with_best"] and best_model_path.is_file():
            checkpoint = torch.load(str(best_model_path))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.flatten_parameters()
            # load optimizer
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.debug(f"Continuing training with best model from bucket {bucket_id}.")

        # PREPARE TRAINING DATA

        bucket_dataset = RecognitionDataset(*train_buckets[bucket_id], alphabet_size=alphabet_size, output_class_num=output_class_num)

        # Variables for early stopping
        epochs_no_improve = 0
        best_epoch = 0

        total_epochs = 0
        running_loss = 0.0
        running_corrects = 0.0
        counter_samples = 0
        current_correct = 0

        relevant_keys = list(relevant_dict.keys())

        # TRAINING
        for batch_idx in range(updates_per_bucket[bucket_id]):
            # CREATE A MINI BATCH
            # select random samples from one length if strategy is single
            if params["training_strategy"] == "single":
                # sampling either a length or a set of states
                key = relevant_keys[np.random.randint(len(relevant_keys))]
                subset = relevant_dict[key]
                idxs = np.random.randint(len(subset), size=model_params[params["model"]]["batch_size"])
                random_indices = subset[idxs]
            elif params["training_strategy"] == "uniform":
                random_indices = None
                # draw for each unit (length or set of states) uniformly at random
                for i, length in enumerate(relevant_dict.keys()):
                    length_indices = relevant_dict[length]
                    random_length_indices = np.random.choice(length_indices, int(samples_per_unit)+1, replace=True)
                    if i == 0:
                        random_indices = random_length_indices
                    else:
                        random_indices = np.concatenate((random_indices, random_length_indices))
                # sample batch_size samples from random_indices
                random_indices = np.random.choice(random_indices, model_params[params["model"]]["batch_size"], replace=True)
            else:
                random_indices = np.random.choice(len(bucket_dataset), model_params[params["model"]]["batch_size"], replace=True)

            inputs, targets, lengths, out_lengths, alphabet_size, num_output_classes = select_collate[params['model']](bucket_dataset[random_indices])
            # get the max over outlengths as integer
            max_out_length = torch.max(out_lengths).type(torch.int).item()
            model.train()
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            out_lengths = out_lengths.to(device)
            optimizer.zero_grad()
            output = model(inputs, max_out_length=max_out_length, lengths=lengths, out_lengths=out_lengths)

            targets = targets[:, :max_out_length]

            loss = loss_fn(output, targets)
            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(output, 2)
            # get argmax of targets
            labels = torch.argmax(targets, 2)
            final = predicted - labels
            # Create a mask where indices are greater than or equal to bucket_dataset.out_lengths
            mask = torch.arange(final.size(1), device=device).unsqueeze(0) >= out_lengths.unsqueeze(1)
            # Apply mask to final
            final[mask] = -1
            # Count zeros, i.e. the positions where predicted == labels
            current_correct = torch.sum(final == 0)

            running_corrects += current_correct.item()
            total_out = torch.sum(out_lengths)
            counter_samples += total_out
            total_batches += 1

            # use the scheduler if it is set
            if scheduler is not None:
                scheduler.step()

            # EVALUATION AND LOGGING
            if batch_idx == 0 or (batch_idx + 1) % params["eval_every_n_batches"] == 0 or (batch_idx + 1) == updates_per_bucket[bucket_id]:
                total_epochs += 1
                train_loss = running_loss / counter_samples.item()
                current_loss = loss / total_out.item()
                train_acc = running_corrects / counter_samples.item()
                current_train_acc = current_correct.item() / total_out.item()
                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)
                writer.add_scalar("loss/train", train_loss, total_batches)
                writer.add_scalar("acc/train", train_acc, total_batches)
                writer.add_scalar("acc/current_train", current_train_acc, total_batches)
                writer.add_scalar("loss/current_train", current_loss, total_batches)

                val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_accuracy)
                writer.add_scalar("loss/val", val_loss, total_batches)
                writer.add_scalar("acc/val", val_accuracy, total_batches)

                unseen_loss, unseen_acc = evaluate(model, unseen_loader, loss_fn, device)
                results["unseen_loss"].append(unseen_loss)
                results["unseen_acc"].append(unseen_acc)
                writer.add_scalar("loss/unseen", unseen_loss, total_batches)
                writer.add_scalar("acc/unseen", unseen_acc, total_batches)
                writer.add_scalar("update", total_batches)



                # EARLY STOPPING
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_accuracy
                    best_epoch = total_epochs
                    # save model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, str(best_model_path))
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if params.get("early_stopping", None):
                        if epochs_no_improve >= params["early_stopping"]:
                            break

                logging.info(
                    f"Bucket {bucket_id} , batch {batch_idx} -- "
                    f"train acc: {round(results['train_acc'][-1], 4)}, "
                    f"val acc: {round(results['val_acc'][-1], 4)}, "
                    f"unseen acc {round(results['unseen_acc'][-1], 4)}, "
                    f"best val acc of {round(best_val_acc, 4)} @ epoch {best_epoch}")

        # END OF LOOP OVER ONE BUCKET
        # copy best model to str(experiment_path / f"{bucket_id}-model.pt")
        # os.system(f"cp {best_model_path} {str(experiment_path / f'{bucket_id}-model.pt')}")
        random_baseline = calculate_random_baseline_from_subset(bucket_dataset)
        results['random_baseline'].append(random_baseline)
        results['updates_per_bucket'].append(batch_idx + 1)
        end_time = time.time()
        results["times"].append(end_time - start_time)


        # Reset the early stopping variables
        epochs_no_improve = 0
        best_epoch = 0

        not_right = {key: len(values) for key, values in results.items() if len(values) != total_epochs and key in
                     ["train_loss", "val_loss", "unseen_loss"]}
        if not_right:
            logging.debug(f"Lengths of results do not match the total number of batches ({total_batches}): {not_right}")

    # END OF TRAINING
    logging.info("Training finished.")
    # Append metrics and log metrics to tensorboard after each epoch
    results["total_updates"] = total_batches

    if total_batches > params["num_updates"]:
        logging.debug(f"Total number of updates exceeded the specified number of updates: {total_batches} > {params['num_updates']}")

    writer.add_scalar("runtime", sum(results["times"]))
    results["runtime"] = [sum(results["times"])]

    # save used datasets
    if config["save_datasets"]:
        torch.save((train_ins, train_outs), str(train_dataset_path))
        torch.save(val_dataset, str(val_dataset_path))
    # best model to final model
    os.system(f"cp {best_model_path} {str(experiment_path / 'final_model.pt')}")

    # DELETE INTERMEDIATE CHECKPOINTS TO SAVE SPACE
    if best_model_path.is_file():
        os.remove(str(best_model_path))
        logging.info(f"Deleted intermediate checkpoint: {best_model_path}")
    # load the final model
    final_checkpoint = torch.load(str(experiment_path / 'final_model.pt'))
    final_model = select_model[params["model"]](**model_params[params["model"]], **params["model_params"])
    final_model.load_state_dict(final_checkpoint['model_state_dict'])
    final_model.to(device)

    data_loader = DataLoader(val_dataset,
                             batch_size=eval_batch_size,
                             collate_fn=select_collate[params['model']],
                             shuffle=False)
    loss, acc = evaluate(final_model, data_loader, loss_fn, device)
    results["final-val_loss"].append(loss)
    results["final-val_acc"].append(acc)
    writer.add_scalar("loss/final-val", loss, total_batches)
    writer.add_scalar("acc/final-val", acc, total_batches)

    if unseen_dataset is not None:
        data_loader = DataLoader(unseen_dataset,
                                 batch_size=eval_batch_size,
                                 collate_fn=select_collate[params['model']],
                                 shuffle=False)
        loss, acc = evaluate(final_model, data_loader, loss_fn, device)
        results["final-unseen_loss"].append(loss)
        results["final-unseen_acc"].append(acc)
        writer.add_scalar("loss/final-unseen", loss, total_batches)
        writer.add_scalar("acc/final-unseen", acc, total_batches)

    # Save metrics
    with open(experiment_path / "results.json", "w") as f:
        json.dump(results, f)

    # Save parameters
    # if provided, are not serializable into json, saved separately anyway
    params.pop("val_loss", None)
    params.pop("val_acc", None)

    params.pop("train_data", None)
    params.pop("val_data", None)
    params.pop("unseen_data", None)
    params.pop("test_data", None)
    params.pop("train_compression", None)
    params.pop("val_compression", None)
    params.pop("test_compression", None)
    params.pop("unseen_compression", None)
    params.pop("language", None)
    with open(str(params_path), "w") as f:
        json.dump(params, f)

    writer.close()
