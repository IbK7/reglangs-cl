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
                    nn.init.constant_(param, 3e-5)  # or some small constant value
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
            # flatten last two dimensions of out
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
            # flatten last two dimensions of out
            out = self.output_activation(self.fc1(hidden[-1]))
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
        out = x.view(x.size(0), -1)
        out = self.activation(self.fc1(out))
        out = self.output_activation(self.fc2(out))
        return out


def train_model(model, hyperparameter, root_path, updates, save_to_file, columns, problem):
    if model != 'transformer':
        expression_length, hidden_size, num_layers, seed, weight_init, learning_rate, nonlinearity, output_activation, batch_size,loss_function = hyperparameter
        file_name = f"{model}_{problem}_length_{expression_length}_hidden_size_{hidden_size}_num_layers_{num_layers}_seed_{seed}_weight_init_{weight_init}_learning_rate_{learning_rate}_batch_size_{batch_size}_nonlinearity_{nonlinearity}_output_activation_{output_activation}_loss_function_{loss_function}.csv"
    else:
        raise NotImplementedError("Transformer model training not implemented yet")

    # check whether the file already exists and has at least two lines
    if Path(root_path / file_name).exists():
        with open(root_path / file_name, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                return

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
        my_model = MyRNN(input_length=data.input_length, input_states=data.num_states, output_size=data.output_classes, hidden_size=hidden_size, nonlinearity=nonlinearity, num_layers=num_layers,
                       seed=seed, weight_init=weight_init, output_activation=output_activation)
    elif model == 'lstm':
        my_model = MyLSTM(input_length=data.input_length, input_states=data.num_states, output_size=data.output_classes, hidden_size=hidden_size, num_layers=num_layers,
                       seed=seed, weight_init=weight_init, output_activation=output_activation)
    elif model == 'transformer':
        raise NotImplementedError("Transformer model training not implemented yet")
    else:
        raise(ValueError(f"Model {model} not recognized"))

    my_model.to(device)

    # train the model on the data m times
    if loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Loss function {loss_function} not recognized")
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    # divide updates by batch size
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
            # save to the big file
            if model == 'transformer':
                raise NotImplementedError("Transformer model training not implemented yet")
            else:
                if correct / len(y) == 1 or i == train_updates - 1:
                    with open(root_path / f"{file_name}", 'a') as f:
                        csv_writer = csv.writer(f)
                        row = f"{expression_length},{hidden_size},{num_layers},{seed},{weight_init},{learning_rate},{batch_size},{nonlinearity},{output_activation},{loss_function},{updates},{train_updates},{i+1},{loss.item()},{correct / len(y)}"
                        csv_writer.writerow(row.split(','))
                    break
    pass


def train(with_torch=True, model='rnn', threads=30):
    result_folder = f"{model}_hyperparameter_search"
    updates = 2**16
    save_to_file = 10
    # hyperparameter configs
    expression_length = [2,3,4, 5, 6, 7, 8]
    hidden_size = [2**i for i in [6,7,8]]
    num_layers = [1]
    batch_size = [2**i for i in [6,7,8]]
    seeds = [42, 43, 44]
    weight_inits = ['default', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform']
    learning_rate = [0.0001, 0.001]
    non_linearities = ['tanh', 'relu']
    output_activations = ['tanh', None,  'relu']
    loss_function = ['cross_entropy', 'mse']
    problems = ['even_pairs', 'cycle_navigation',  'parity_check']

    if model == 'lstm':
        non_linearities = [None]

    # create all hyperparameter combinations using the above lists
    if model == 'transformer':
        raise NotImplementedError("Transformer model training not implemented yet")
    else:
        hyperparameters = list(itertools.product(expression_length, hidden_size, num_layers, seeds, weight_inits, learning_rate, non_linearities, output_activations, batch_size,loss_function))
        all_columns = f'length,hidden_size,num_layers,seed,weight_init,learning_rate,batch_size,nonlinearity,output_activation,loss_function,sample_updates,total_mini_batches,current_mini_batch,loss,accuracy'

    print(f"Number of hyperparameters: {len(hyperparameters)}")

    for problem in problems:
        # print start of problem with timestamp
        print(f"Start training {problem} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        root_path = RESULT_PATH / result_folder / f'{problem}'
        if not Path(root_path).exists():
            Path(root_path).mkdir(parents=True, exist_ok=True)
        # one file for each thread
        joblib.Parallel(n_jobs=threads)(joblib.delayed(train_model)(model, hyperparameter, root_path, updates, save_to_file, columns=all_columns, problem=problem) for hyperparameter in hyperparameters)



@click.command()
@click.option('--with_torch', default=True, type=bool)
@click.option('--model', default='rnn', type=str)
@click.option('--threads', default=1, type=int)
def main(with_torch, model, threads):
    print(f"Training with torch: {with_torch}, model: {model}, threads: {threads}")
    train(with_torch=with_torch, model=model, threads=threads)

if __name__ == "__main__":
    main()
