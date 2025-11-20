import torch
import hashlib
import torch.nn as nn
from pathlib import Path
from collections import defaultdict


def set_training_paths(base_path, params):
    if isinstance(params["experiment_id"], str):
        experiment_id = params["experiment_id"]
    elif isinstance(params["experiment_id"], int):
        experiment_id = str.zfill(str(params['experiment_id']), 6)
    else:
        raise ValueError("experiment_id must be either a string or an integer.")
    # Path for the experiment
    experiment_path = base_path / experiment_id / "_".join(
        [params["model"], str(Path(params['problem']).name)]) / params["run_dir"]
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Path for the model and parameters
    best_model_path = experiment_path / "best_model.pt"
    params_path = experiment_path / "params.json"
    train_dataset_path = experiment_path / "orig_train_dataset.pt"
    val_dataset_path = experiment_path / "orig_val_dataset.pt"

    return experiment_path, best_model_path, params_path, train_dataset_path, val_dataset_path


def collect_and_write_results(results, writer, batches, prefix=None, **kwargs):
    for metric_name, metric_value in kwargs.items():
        if metric_value is not None:
            save_metric_name = f"{prefix}-{metric_name}" if prefix else metric_name
            results[save_metric_name].append(metric_value)
            if metric_name == "batch_loss":
                if prefix == "epoch":
                    continue
                for i, loss in enumerate(metric_value):
                    writer.add_scalar("loss/batch", loss, batches - len(metric_value) + i)
            else:
                writer.add_scalar("/".join(save_metric_name.split("_")[::-1]), metric_value, batches)


def collect_results(results, prefix=None, **kwargs):
    for metric_name, metric_value in kwargs.items():
        save_metric_name = f"{prefix}-{metric_name}" if prefix else metric_name
        results[save_metric_name].append(metric_value)


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device).type(torch.float64)
    total_correct = torch.tensor(0.0).to(device).type(torch.float64)
    total_out = torch.tensor(0.0).to(device).type(torch.float64)
    with torch.no_grad():
        for inputs, targets, lengths, out_lengths, alphabet_size, num_output_classes in data_loader:
            max_output_length = torch.max(out_lengths).type(torch.int).item()
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            out_lengths = out_lengths.to(device)
            targets = targets[:, :max_output_length]
            targets = targets.to(device)
            output = model(inputs, max_out_length=max_output_length, lengths=lengths, out_lengths=out_lengths)
            # Create a mask where indices are greater than or equal to bucket_dataset.out_lengths
            mask = torch.arange(output.size(1), device=device).unsqueeze(0) >= out_lengths.to(device).unsqueeze(1)
            # Apply mask to output
            output[mask] = 0

            loss = loss_fn(output, targets)
            total_loss += loss.type(torch.float64)

            predicted = torch.argmax(output, 2)
            # get argmax of targets
            labels = torch.argmax(targets, 2)
            final = predicted - labels
            # Create a mask where indices are greater than or equal to bucket_dataset.out_lengths
            mask = torch.arange(final.size(1), device=device).unsqueeze(0) >= out_lengths.to(device).unsqueeze(1)
            # Apply mask to final
            final[mask] = -1
            # Count zeros
            current_correct = torch.sum(final == 0)
            total_correct += current_correct
            total_out += torch.sum(out_lengths)
    del output, targets, inputs, labels, final, mask, loss
    torch.cuda.empty_cache()
    return (total_loss.item() / len(data_loader), total_correct.item() / total_out.item())


def collect_hidden_states(model, data_loader, device):
    model.eval()
    hidden_states = []
    with torch.no_grad():
        for inputs, targets, lengths in data_loader:
            inputs = inputs.to(device)
            output, hidden = model(inputs, return_hidden_states=True)
            hidden_states.append(hidden)
    return torch.cat(hidden_states, dim=1)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total = 0.0
    num_batches = len(train_loader)

    # shuffle the bucket indices
    for batch_idx, (inputs, targets, lengths) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == torch.argmax(targets, 1)).sum()
        total += targets.size(0)

    return total_loss / num_batches, total_correct / total


def train_one_greedy_batch(model, batch, loss_fn, optimizer, device):
    """
    Train the model for one batch.
    """
    model.train()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    output = model(inputs)
    loss = loss_fn(output, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(output, 1)

    return loss.item(), (predicted == torch.argmax(targets, 1)).sum().item() / len(targets)


def hash_model_parameters(model):
    model_state_dict = model.state_dict()
    model_parameters_string = str(model_state_dict)
    model_hash = hashlib.md5(model_parameters_string.encode()).hexdigest()
    return model_hash


def reset_to_checkpoint(model, optimizer, path, double_check=False):
    """
    Load the model and optimizer from a checkpoint.
    """
    if double_check:
        prev_model_hash = hash_model_parameters(model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if double_check:
        new_model_hash = hash_model_parameters(model)
        assert prev_model_hash != new_model_hash, "Model parameters are the same after reset."


def compute_training_metrics(model: nn.Module, sketch_size: float = None):
    metrics = defaultdict(list)
    all_weights = []
    all_biases = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.extend(param.flatten())

            l1_norm = torch.norm(param, p=1).item()
            l2_norm = torch.norm(param, p=2).item()
            l1_over_l2 = l1_norm / l2_norm if l2_norm != 0 else 0

            # Calculate SVD for weight matrices if applicable
            if len(param.shape) > 1:  # Only apply SVD to matrices
                if sketch_size is not None:
                    s = torch.linalg.svdvals(matrix_sketching(param, param.shape[0]*sketch_size))
                else:
                    s = torch.linalg.svdvals(param)
                sv_mean = torch.mean(s).item()
                sv_variance = torch.var(s).item()

                trace = torch.trace(param).item()
                spectral_norm = torch.max(s).item()
                trace_over_spectral = trace / spectral_norm if spectral_norm != 0 else 0

            else:
                sv_mean = sv_variance = spectral_norm = trace = trace_over_spectral = 0

            metrics['L1_norm'].append(l1_norm)
            metrics['L2_norm'].append(l2_norm)
            metrics['l1_over_l2_norm'].append(l1_over_l2)
            metrics['trace'].append(trace)
            metrics['singular_values_mean'].append(sv_mean)
            metrics['singular_values_variance'].append(sv_variance)
            metrics['spectral_norm'].append(spectral_norm)
            metrics['trace_over_spectral_norm'].append(trace_over_spectral)

        elif 'bias' in name:
            all_biases.extend(param.flatten())

    # Aggregate the metrics that are averaged over all layers
    aggregated_metrics = {key: sum(values) / len(values) if values else 0 for key, values in metrics.items()}
    aggregated_metrics["median_weights"] = torch.median(torch.tensor(all_weights)).item()
    aggregated_metrics["median_biases"] = torch.median(torch.tensor(all_biases)).item()
    aggregated_metrics["mean_weights"] = torch.mean(torch.tensor(all_weights)).item()
    aggregated_metrics["mean_biases"] = torch.mean(torch.tensor(all_biases)).item()
    aggregated_metrics["variance_weights"] = torch.var(torch.tensor(all_weights)).item()
    aggregated_metrics["variance_biases"] = torch.var(torch.tensor(all_biases)).item()
    return aggregated_metrics


def matrix_sketching(matrix, sketch_size):
    """
    Performs matrix sketching on matrix X.

    Args:
        matrix (torch.Tensor): The input matrix (m x n).
        sketch_size (int): The size of the sketched matrix (k), where k < min(m, n).

    Returns:
        torch.Tensor: The sketched matrix (k x n).
    """
    # Create a random projection matrix
    random_projection = torch.randn(matrix.shape[1], sketch_size)

    # Compute the sketched matrix
    sketched_matrix = torch.matmul(matrix, random_projection)

    return sketched_matrix
