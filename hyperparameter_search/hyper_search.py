# hyper_search.py
import csv
import time
import torch
import click
import joblib
import itertools
import numpy as np
from torch import nn
from pathlib import Path
from typing import Union, Optional
import math
from tqdm import tqdm

from defaultvalues import RESULT_PATH
from hyperparameter_search.test_problems import EvenPairs, ParityCheck, FirstA, CycleNavigation, FirstLastA, BucketSort

class MyRNN(nn.RNN):
    def __init__(self, input_length:int, input_states: int, hidden_size: int, output_size: int, nonlinearity="tanh", num_layers=1, dropout: float = 0.0, seed=42, weight_init='default', output_activation=None):
        torch.manual_seed(seed)
        super().__init__(input_states, hidden_size, nonlinearity=nonlinearity, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.input_size = input_states
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Identity()
        if output_activation is not None:
            if output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            elif output_activation == 'relu':
                self.output_activation = nn.ReLU()
        if weight_init != 'default':
            self.init_weights(weight_init)


    def init_weights(self, weight_init):
        # set torch seed
        torch.manual_seed(10)
        for name, param in self.named_parameters():
            if weight_init == 'xavier_normal':
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'xavier_uniform':
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_normal':
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_uniform':
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)

    def forward(self, x: torch.Tensor, unpack: bool = None, return_hidden_states=False) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hidden = super().forward(x)
        if return_hidden_states:
            return out, hidden
        else:
            out = self.output_activation(self.fc1(hidden[-1]))
        return out

class MyLSTM(nn.LSTM):
    def __init__(self, input_length:int, input_states: int, hidden_size: int, output_size: int, num_layers=1, dropout: float = 0.0, seed=42, weight_init='default', output_activation=None):
        torch.manual_seed(seed)
        super().__init__(input_states, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.input_size = input_states
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Identity()
        if output_activation is not None:
            if output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            elif output_activation == 'relu':
                self.output_activation = nn.ReLU()
        if weight_init != 'default':
            self.init_weights(weight_init)

    def init_weights(self, weight_init):
        # set torch seed
        torch.manual_seed(10)
        for name, param in self.named_parameters():
            if weight_init == 'xavier_normal':
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'xavier_uniform':
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_normal':
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_uniform':
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)

    def forward(self, x: torch.Tensor, unpack: bool = None, return_hidden_states=False) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, (hidden, cell) = super().forward(x)
        if return_hidden_states:
            return out, hidden
        else:
            out = self.output_activation(self.fc1(hidden[-1]))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self, input_length: int, input_states: int, hidden_size: int, output_size: int, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.0, 
                 seed: int = 42, weight_init: str = 'default', output_activation=None):
        torch.manual_seed(seed)
        super().__init__()
        self.input_size = input_states
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Project input to model dimension
        self.input_proj = nn.Linear(input_states, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer for classification
        self.fc1 = nn.Linear(hidden_size, output_size)
        
        self.output_activation = nn.Identity()
        if output_activation is not None:
            if output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            elif output_activation == 'relu':
                self.output_activation = nn.ReLU()
        
        if weight_init != 'default':
            self.init_weights(weight_init)

    def init_weights(self, weight_init):
        torch.manual_seed(10)
        for name, param in self.named_parameters():
            if weight_init == 'xavier_normal':
                if 'weight' in name:
                    if param.dim() >= 2:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'xavier_uniform':
                if 'weight' in name:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_normal':
                if 'weight' in name:
                    if param.dim() >= 2:
                        nn.init.kaiming_normal_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_uniform':
                if 'weight' in name:
                    if param.dim() >= 2:
                        nn.init.kaiming_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)

    def forward(self, x: torch.Tensor, unpack: bool = None, return_hidden_states=False) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [batch, seq_len, input_states]
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        
        if return_hidden_states:
            return h, h[:, -1, :].unsqueeze(0)
        else:
            # Use last timestep for classification (like RNN/LSTM)
            out = self.output_activation(self.fc1(h[:, -1, :]))
        return out


class MyFC(nn.Module):
    def __init__(self, input_length:int, input_states: int, hidden_size: int, output_size: int, nonlinearity="tanh", num_layers=1, dropout: float = 0.0, seed=42, weight_init='default', output_activation=None):
        torch.manual_seed(seed)
        super().__init__()
        self.input_size = input_states
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_states*input_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        if nonlinearity == "tanh":
            self.activation = nn.Tanh()
        elif nonlinearity == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        if output_activation is not None:
            if output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            elif output_activation == 'relu':
                self.output_activation = nn.ReLU()
        self.output_activation = nn.Identity()
        if nonlinearity == "tanh":
            self.output_activation = nn.Tanh()
        elif nonlinearity == "relu":
            self.output_activation = nn.ReLU()
        if weight_init != 'default':
            self.init_weights(weight_init)

    def init_weights(self, weight_init):
        torch.manual_seed(10)
        for name, param in self.named_parameters():
            if weight_init == 'xavier_normal':
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'xavier_uniform':
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_normal':
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)
            elif weight_init == 'kaiming_uniform':
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 3e-5)

    def forward(self, x: torch.Tensor, unpack: bool = None, return_hidden_states=False) -> Union[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out = x.view(x.size(0), -1)
        out = self.activation(self.fc1(out))
        out = self.output_activation(self.fc2(out))
        return out


def train_model(model, hyperparameter, root_path, updates, save_to_file, columns, problem):
    if model == 'transformer':
        expression_length, hidden_size, num_heads, num_layers, seed, weight_init, learning_rate, output_activation, batch_size, dropout, loss_function = hyperparameter
        file_name = f"{model}_{problem}_length_{expression_length}_hidden_size_{hidden_size}_num_heads_{num_heads}_num_layers_{num_layers}_seed_{seed}_weight_init_{weight_init}_learning_rate_{learning_rate}_batch_size_{batch_size}_output_activation_{output_activation}_dropout_{dropout}_loss_function_{loss_function}.csv"
    else:
        expression_length, hidden_size, num_layers, seed, weight_init, learning_rate, nonlinearity, output_activation, batch_size, loss_function = hyperparameter
        file_name = f"{model}_{problem}_length_{expression_length}_hidden_size_{hidden_size}_num_layers_{num_layers}_seed_{seed}_weight_init_{weight_init}_learning_rate_{learning_rate}_batch_size_{batch_size}_nonlinearity_{nonlinearity}_output_activation_{output_activation}_loss_function_{loss_function}.csv"

    # check whether the file already exists and has at least two lines
    if Path(root_path / file_name).exists():
        with open(root_path / file_name, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                return 'skipped'

    device = 'cpu'
    if problem == 'even_pairs':
        data = EvenPairs(expression_length, device=device)
    elif problem == 'parity_check':
        data = ParityCheck(expression_length, device=device)
    elif problem == 'first_a':
        data = FirstA(expression_length, device=device)
    elif problem == 'cycle_navigation':
        data = CycleNavigation(expression_length, cycle_length=5, device=device)
    elif problem == 'cycle_navigation_small':
        data = CycleNavigation(expression_length, cycle_length=3, device=device)
    elif problem == 'first_last_a':
        data = FirstLastA(expression_length, device=device)
    elif problem == 'bucket_sort':
        data = BucketSort(expression_length, device=device)
    else:
        raise ValueError(f"Problem {problem} not recognized")

    with open(root_path / file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(columns.split(','))

    if model == 'rnn':
        my_model = MyRNN(input_length=data.input_length, input_states=data.num_states, output_size=data.output_classes, 
                         hidden_size=hidden_size, nonlinearity=nonlinearity, num_layers=num_layers,
                         seed=seed, weight_init=weight_init, output_activation=output_activation)
    elif model == 'lstm':
        my_model = MyLSTM(input_length=data.input_length, input_states=data.num_states, output_size=data.output_classes, 
                          hidden_size=hidden_size, num_layers=num_layers,
                          seed=seed, weight_init=weight_init, output_activation=output_activation)
    elif model == 'transformer':
        my_model = MyTransformer(input_length=data.input_length, input_states=data.num_states, output_size=data.output_classes,
                                 hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                                 seed=seed, weight_init=weight_init, output_activation=output_activation)
    else:
        raise ValueError(f"Model {model} not recognized")

    my_model.to(device)

    # train the model on the data
    if loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Loss function {loss_function} not recognized")
    
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    train_updates = updates // batch_size
    
    for i in range(train_updates):
        rand_indices = np.random.choice(len(data.X), batch_size, replace=True)
        x = data.X[rand_indices]
        y = data.y[rand_indices]

        optimizer.zero_grad()
        output = my_model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if i % save_to_file == 0 or i == train_updates - 1:
            predicted = torch.argmax(output, 1)
            true_labels = torch.argmax(y, 1)
            correct = (predicted == true_labels).sum().item()
            
            if model == 'transformer':
                if correct / len(y) == 1 or i == train_updates - 1:
                    with open(root_path / f"{file_name}", 'a') as f:
                        csv_writer = csv.writer(f)
                        row = f"{expression_length},{hidden_size},{num_heads},{num_layers},{seed},{weight_init},{learning_rate},{batch_size},{output_activation},{dropout},{loss_function},{updates},{train_updates},{i+1},{loss.item()},{correct / len(y)}"
                        csv_writer.writerow(row.split(','))
                    return 'completed'
            else:
                if correct / len(y) == 1 or i == train_updates - 1:
                    with open(root_path / f"{file_name}", 'a') as f:
                        csv_writer = csv.writer(f)
                        row = f"{expression_length},{hidden_size},{num_layers},{seed},{weight_init},{learning_rate},{batch_size},{nonlinearity},{output_activation},{loss_function},{updates},{train_updates},{i+1},{loss.item()},{correct / len(y)}"
                        csv_writer.writerow(row.split(','))
                    return 'completed'
    
    return 'completed'


def train(with_torch=True, model='rnn', threads=30):
    result_folder = f"{model}_hyperparameter_search"
    updates = 2**16
    save_to_file = 10
    
    # hyperparameter config
    expression_length = [2, 3, 4, 5] 
    seeds = [42]  # Reduced from [42, 43, 44]
    weight_inits = ['kaiming_uniform', 'xavier_uniform'] 
    learning_rate = [0.0001, 0.001]  
    output_activations = ['relu', 'tanh']  # Reduced from ['tanh', None, 'relu']
    loss_function = ['cross_entropy']  # Reduced from ['cross_entropy', 'mse']
    problems = ['cycle_navigation', 'even_pairs', 'first_a']#, 'first_last_a', 'bucket_sort']  
    
    if model == 'transformer':
        # Transformer-specific hyperparameters - REDUCED FOR TESTING
        hidden_size = [64, 128]  # Reduced from [64, 128, 256]
        num_heads = [4, 8]  # Kept same (already minimal)
        num_layers = [3, 5]  # Reduced from [2, 3, 4]
        batch_size = [63, 128]  # Reduced from [64, 128, 256]
        dropout = [0.0, 0.1]  # Reduced from [0.0, 0.1]
        
        hyperparameters = list(itertools.product(
            expression_length, hidden_size, num_heads, num_layers, seeds, 
            weight_inits, learning_rate, output_activations, batch_size, 
            dropout, loss_function
        ))
        all_columns = 'length,hidden_size,num_heads,num_layers,seed,weight_init,learning_rate,batch_size,output_activation,dropout,loss_function,sample_updates,total_mini_batches,current_mini_batch,loss,accuracy'
    else:
        # RNN/LSTM hyperparameters - REDUCED FOR TESTING
        hidden_size = [128, 256]  # Reduced from [64, 128, 256]
        num_layers = [1]
        batch_size = [128]  # Reduced from [64, 128, 256]
        non_linearities = ['tanh']  # Reduced from ['tanh', 'relu']
        
        if model == 'lstm':
            non_linearities = [None]
        
        hyperparameters = list(itertools.product(
            expression_length, hidden_size, num_layers, seeds, weight_inits, 
            learning_rate, non_linearities, output_activations, batch_size, 
            loss_function
        ))
        all_columns = 'length,hidden_size,num_layers,seed,weight_init,learning_rate,batch_size,nonlinearity,output_activation,loss_function,sample_updates,total_mini_batches,current_mini_batch,loss,accuracy'

    total_configs = len(hyperparameters) * len(problems)
    print(f"Number of hyperparameter configurations per problem: {len(hyperparameters)}")
    print(f"Total configurations across all problems: {total_configs}")
    print(f"Using {threads} threads")
    print("-" * 80)

    overall_start_time = time.time()
    
    for problem_idx, problem in enumerate(problems):
        print(f"\n[Problem {problem_idx + 1}/{len(problems)}] Starting {problem} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        root_path = RESULT_PATH / result_folder / f'{problem}'
        if not Path(root_path).exists():
            Path(root_path).mkdir(parents=True, exist_ok=True)
        
        problem_start_time = time.time()
        
        # Use joblib with progress tracking via return values
        with tqdm(total=len(hyperparameters), desc=f"{problem}", unit="config") as pbar:
            # Process in batches to update progress bar
            results = joblib.Parallel(n_jobs=threads, return_as='generator')(
                joblib.delayed(train_model)(model, hyperparameter, root_path, updates, save_to_file, columns=all_columns, problem=problem) 
                for hyperparameter in hyperparameters
            )
            
            skipped = 0
            completed = 0
            for result in results:
                if result == 'skipped':
                    skipped += 1
                else:
                    completed += 1
                pbar.update(1)
                pbar.set_postfix({'completed': completed, 'skipped': skipped})
        
        problem_elapsed = time.time() - problem_start_time
        print(f"Completed {problem} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"  Time: {problem_elapsed/60:.1f} minutes | Completed: {completed} | Skipped: {skipped}")
    
    overall_elapsed = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print(f"All hyperparameter searches completed!")
    print(f"Total time: {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.2f} hours)")
    print(f"Results saved in: {RESULT_PATH / result_folder}")
    print("=" * 80)


@click.command()
@click.option('--with_torch', default=True, type=bool)
@click.option('--model', default='rnn', type=str)
@click.option('--threads', default=1, type=int)
def main(with_torch, model, threads):
    print(f"Training with torch: {with_torch}, model: {model}, threads: {threads}")
    train(with_torch=with_torch, model=model, threads=threads)

if __name__ == "__main__":
    main()