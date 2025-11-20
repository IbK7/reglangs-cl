import abc
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as nn


class Problem(abc.ABC):
    """
    Baseclass for problems that are not representable by automata
    """
    def __init__(self,
                 name: str,
                 symbols=None,
                 num_train_samples=1000,
                    num_unseen_samples=1000,
                 min_train_input_length=1,
                 max_train_input_length=40,
                 min_unseen_input_length=41,
                 max_unseen_input_length=500,
                 min_train_output_length=1,
                 max_train_output_length=40,
                    min_unseen_output_length=41,
                 max_unseen_output_length=500):
        self.name = name
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.symbol_representation = {symbol: i for i, symbol in enumerate(symbols)}
        self.num_samples = num_train_samples
        self.num_unseen_samples = num_unseen_samples
        self.min_train_input_length = min_train_input_length
        self.max_train_input_length = max_train_input_length
        self.min_unseen_input_length = min_unseen_input_length
        self.max_unseen_input_length = max_unseen_input_length
        self.min_train_output_length = min_train_output_length
        self.max_train_output_length = max_train_output_length
        self.min_unseen_output_length = min_unseen_output_length
        self.max_unseen_output_length = max_unseen_output_length
        self.training_samples_per_length = {}
        self.unseen_samples_per_length = {}

        self.X, self.y, self.lengths_X, self.lengths_y = self.generate_train_data()
        self.unseen_X, self.unseen_y, self.lengths_unseen_X, self.lengths_unseen_y = self.generate_unseen_data()




    @abc.abstractmethod
    def generate_data_per_length(self, num_samples: int, length) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate data
        """

    def generate_data(self, num_samples: int, min_length:int, max_length:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate data
        """
        X = None
        y = None
        l_X = None
        l_y = None
        samples_per_length = num_samples // (max_length - min_length + 1)
        for length in range(min_length, max_length + 1):
            X_, y_ = self.generate_data_per_length(samples_per_length, length)
            l_X_ = torch.full((X_.shape[0],), X_.shape[1], dtype=torch.int32)
            l_y_ = torch.full((y_.shape[0],), y_.shape[1], dtype=torch.int32)
            # add padding with -1
            padding_input_length = self.max_train_input_length if length <= self.max_train_input_length else self.max_unseen_input_length
            padding_output_length = self.max_train_output_length if length <= self.max_train_output_length else self.max_unseen_output_length
            X_ = nn.pad(X_, (0, padding_input_length - X_.shape[1]), value=-1)
            y_ = nn.pad(y_, (0, padding_output_length - y_.shape[1]), value=-1)

            if X is None:
                X = X_
                y = y_
                l_X = l_X_
                l_y = l_y_
            else:
                X = torch.cat((X, X_), dim=0)
                y = torch.cat((y, y_), dim=0)
                l_X = torch.cat((l_X, l_X_), dim=0)
                l_y = torch.cat((l_y, l_y_), dim=0)
        return X, y, l_X, l_y


    def generate_train_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training data
        """
        X, y, l_x, l_y = self.generate_data(self.num_samples, self.min_train_input_length, self.max_train_input_length)
        # unique lengths with counts
        unique, counts = torch.unique(l_x, return_counts=True)
        self.training_samples_per_length = dict(zip(unique.tolist(), counts.tolist()))
        return X, y, l_x, l_y



    def generate_unseen_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate unseen data
        """
        X, y, l_x, l_y = self.generate_data(self.num_unseen_samples, self.min_unseen_input_length, self.max_unseen_input_length)
        # unique lengths with counts
        unique, counts = torch.unique(l_x, return_counts=True)
        self.unseen_samples_per_length = dict(zip(unique.tolist(), counts.tolist()))
        return X, y, l_x, l_y



class ReverseString(Problem):
    """
    Problem where the output is the reverse of the input
    """
    def __init__(self,
                 symbols=None,
                 num_train_samples=1000,
                    num_unseen_samples=1000,
                 min_train_input_length=1,
                    max_train_input_length=40,
                    min_unseen_input_length=41,
                    max_unseen_input_length=500):


        if symbols is None:
            symbols = ['a', 'b']
        super().__init__(name="reverse_string",
                         symbols=symbols,
                            num_train_samples=num_train_samples,
                            num_unseen_samples=num_unseen_samples,
                            min_train_input_length=min_train_input_length,
                            max_train_input_length=max_train_input_length,
                            min_unseen_input_length=min_unseen_input_length,
                            max_unseen_input_length=max_unseen_input_length,
                            min_train_output_length=min_train_input_length,
                            max_train_output_length=max_train_input_length,
                            min_unseen_output_length=min_unseen_input_length,
                            max_unseen_output_length=max_unseen_input_length)


    def generate_data_per_length(self, num_samples: int, length) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data
        """
        if self.num_symbols**length < num_samples:
            # generate all possible sequences
            X = torch.Tensor(np.array(np.meshgrid(*[range(self.num_symbols) for _ in range(length)])).T.reshape(-1, length))
            y = torch.flip(X, dims=[1])
        else:
            X = torch.Tensor(np.random.choice(self.num_symbols, size=(num_samples, length)))
            y = torch.flip(X, dims=[1])
        return X, y


class BucketSort(Problem):
    """
    Problem where the output is the reverse of the input
    """
    def __init__(self,
                 symbols=5,
                 num_train_samples=1000,
                    num_unseen_samples=1000,
                 min_train_input_length=1,
                    max_train_input_length=40,
                    min_unseen_input_length=41,
                    max_unseen_input_length=500):


        if symbols is None:
            symbols = ['a', 'b']
        else:
            symbols = [str(i) for i in range(symbols)]
        super().__init__(name="bucket_sort",
                         symbols=symbols,
                            num_train_samples=num_train_samples,
                            num_unseen_samples=num_unseen_samples,
                            min_train_input_length=min_train_input_length,
                            max_train_input_length=max_train_input_length,
                            min_unseen_input_length=min_unseen_input_length,
                            max_unseen_input_length=max_unseen_input_length,
                            min_train_output_length=min_train_input_length,
                            max_train_output_length=max_train_input_length,
                            min_unseen_output_length=min_unseen_input_length,
                            max_unseen_output_length=max_unseen_input_length)


    def generate_data_per_length(self, num_samples: int, length) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data
        """
        if self.num_symbols**length < num_samples:
            # generate all possible sequences
            X = torch.Tensor(np.array(np.meshgrid(*[range(self.num_symbols) for _ in range(length)])).T.reshape(-1, length))
            # sort
            y = torch.sort(X, dim=1).values
        else:
            X = torch.Tensor(np.random.choice(self.num_symbols, size=(num_samples, length)))
            y = torch.sort(X, dim=1).values
        return X, y


class BucketSort(Problem):
    """
    Problem where the output is the sorted input
    """
    def __init__(self,
                 symbols=None,
                 num_train_samples=1000,
                    num_unseen_samples=1000,
                 min_train_input_length=1,
                    max_train_input_length=40,
                    min_unseen_input_length=41,
                    max_unseen_input_length=500):


        if symbols is None:
            symbols = [0, 1, 2, 3, 4]
        super().__init__(name="bucket_sort",
                         symbols=symbols,
                            num_train_samples=num_train_samples,
                            num_unseen_samples=num_unseen_samples,
                            min_train_input_length=min_train_input_length,
                            max_train_input_length=max_train_input_length,
                            min_unseen_input_length=min_unseen_input_length,
                            max_unseen_input_length=max_unseen_input_length,
                            min_train_output_length=min_train_input_length,
                            max_train_output_length=max_train_input_length,
                            min_unseen_output_length=min_unseen_input_length,
                            max_unseen_output_length=max_unseen_input_length)


    def generate_data_per_length(self, num_samples: int, length) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data
        """
        if self.num_symbols**length < num_samples:
            # generate all possible sequences
            X = torch.Tensor(np.array(np.meshgrid(*[range(self.num_symbols) for _ in range(length)])).T.reshape(-1, length))
            # y is the sorted version of X
            y = torch.sort(X, dim=1).values
        else:
            X = torch.Tensor(np.random.choice(self.num_symbols, size=(num_samples, length)))
            y = torch.sort(X, dim=1).values
        return X, y


