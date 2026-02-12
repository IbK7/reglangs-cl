# test_problem.py
import abc
from abc import abstractmethod

import torch


class Problem(abc.ABC):
    def __init__(self, name, num_states, output_classes, input_length,  device='cpu'):
        self.name = name
        self.X = None
        self.y = None
        self.num_states = num_states
        self.output_classes = output_classes
        self.input_length = input_length
        self.device = device
        self.set_input_output()


    # override this method in the child class
    @abstractmethod
    def set_input_output(self):
        pass


class EvenPairs(Problem):
    def __init__(self, input_length, device):
        super().__init__('even_pairs', num_states=2, output_classes=2, input_length=input_length, device=device)

    def set_input_output(self):
        # get all binary combinations of 0 and 1 of length n
        combinations = []
        for i in range(2 ** self.input_length):
            binary = bin(i)[2:]
            binary = '0' * (self.input_length - len(binary)) + binary
            combinations.append(list(map(int, binary)))
        # one-hot encode the combinations
        one_hot_combinations = []
        y = torch.zeros(len(combinations), dtype=torch.long)
        for i, combination in enumerate(combinations):
            one_hot_combinations.append(torch.nn.functional.one_hot(torch.tensor(combination), num_classes=2).float())
            label = combination[0] == combination[-1]
            y[i] = label
        # get tensor
        x = torch.stack(one_hot_combinations)
        # one hot y
        y = torch.nn.functional.one_hot(y, num_classes=self.output_classes).float()
        self.X = x.to(self.device)
        self.y = y.to(self.device)

class FirstA(Problem):
    def __init__(self, input_length, device):
        super().__init__('first_a', num_states=2, output_classes=2, input_length=input_length, device=device)

    def set_input_output(self):
        # get all binary combinations of 0 and 1 of length n
        combinations = []
        for i in range(2 ** self.input_length):
            binary = bin(i)[2:]
            binary = '0' * (self.input_length - len(binary)) + binary
            combinations.append(list(map(int, binary)))
        # one-hot encode the combinations
        one_hot_combinations = []
        y = torch.zeros(len(combinations), dtype=torch.long)
        for i, combination in enumerate(combinations):
            one_hot_combinations.append(torch.nn.functional.one_hot(torch.tensor(combination), num_classes=self.output_classes).float())
            label = combination[0] == 0
            y[i] = label
        # get tensor
        x = torch.stack(one_hot_combinations)
        # one hot y
        y = torch.nn.functional.one_hot(y, num_classes=self.output_classes).float()
        self.X = x.to(self.device)
        self.y = y.to(self.device)


class FirstLastA(Problem):
    def __init__(self, input_length, device):
        super().__init__('first_last_a', num_states=2, output_classes=2, input_length=input_length, device=device)

    def set_input_output(self):
        # get all binary combinations of 0 and 1 of length n
        combinations = []
        for i in range(2 ** self.input_length):
            binary = bin(i)[2:]
            binary = '0' * (self.input_length - len(binary)) + binary
            combinations.append(list(map(int, binary)))
        # one-hot encode the combinations
        one_hot_combinations = []
        y = torch.zeros(len(combinations), dtype=torch.long)
        for i, combination in enumerate(combinations):
            one_hot_combinations.append(torch.nn.functional.one_hot(torch.tensor(combination), num_classes=self.output_classes).float())
            label = combination[0] == combination[-1] == 0
            y[i] = label
        # get tensor
        x = torch.stack(one_hot_combinations)
        # one hot y
        y = torch.nn.functional.one_hot(y, num_classes=self.output_classes).float()
        self.X = x.to(self.device)
        self.y = y.to(self.device)

class ParityCheck(Problem):
    def __init__(self, input_length, device):
        super().__init__('parity_check', num_states=2, output_classes=2, input_length=input_length, device=device)

    def set_input_output(self):
        # get all binary combinations of 0 and 1 of length n
        combinations = []
        for i in range(2 ** self.input_length):
            binary = bin(i)[2:]
            binary = '0' * (self.input_length - len(binary)) + binary
            combinations.append(list(map(int, binary)))
        # one-hot encode the combinations
        one_hot_combinations = []
        y = torch.zeros(len(combinations), dtype=torch.long)
        for i, combination in enumerate(combinations):
            one_hot_combinations.append(torch.nn.functional.one_hot(torch.tensor(combination), num_classes=self.output_classes).float())
            label = sum(combination) % 2 == 0
            y[i] = label
        # get tensor
        x = torch.stack(one_hot_combinations)
        # one hot y
        y = torch.nn.functional.one_hot(y, num_classes=self.output_classes).float()
        self.X = x.to(self.device)
        self.y = y.to(self.device)


def base_10_to_base_3(i, input_length):
    ternary = []
    while i > 0:
        ternary.insert(0, i % 3)
        i = i // 3
    return [0] * (input_length - len(ternary)) + ternary


class CycleNavigation(Problem):
    def __init__(self,  input_length, cycle_length=5, device='cpu'):
        self.cycle_length = cycle_length
        super().__init__('cycle_navigation', num_states=3, output_classes=cycle_length, input_length=input_length, device=device)


    def set_input_output(self):
        # get all combinations of 0,1,2 of length n
        combinations = []
        for i in range(3 ** self.input_length):
            ternary = base_10_to_base_3(i, self.input_length)
            combinations.append(ternary)

        # one-hot encode the combinations
        one_hot_combinations = []
        y = torch.zeros(len(combinations), dtype=torch.long)
        for i, combination in enumerate(combinations):
            one_hot_combinations.append(torch.nn.functional.one_hot(torch.tensor(combination), num_classes=3).float())
            label = 0
            for j in combination:
                if j == 0:
                    label += 1
                    label = label % self.cycle_length
                elif j == 1:
                    label -= 1
                    label = label % self.cycle_length
                else:
                    continue
            y[i] = label
        # get tensor
        x = torch.stack(one_hot_combinations)
        # one hot y
        y = torch.nn.functional.one_hot(y, num_classes=self.cycle_length).float()
        self.X = x.to(self.device)
        self.y = y.to(self.device)

class BucketSort(Problem):
    def __init__(self, input_length, device='cpu'):
        super().__init__('bucket_sort', num_states=10, output_classes=10, input_length=input_length, device=device)

    def set_input_output(self):
        combinations = []
        for i in range(10 ** self.input_length):
            digits = [int(d) for d in str(i).zfill(self.input_length)]
            combinations.append(digits)

        one_hot_combinations = []
        sorted_outputs = []
        for digits in combinations:
            one_hot_combinations.append(torch.nn.functional.one_hot(torch.tensor(digits), num_classes=10).float())
            sorted_digits = sorted(digits)
            sorted_outputs.append(torch.tensor(sorted_digits).unsqueeze(0))

        x = torch.stack(one_hot_combinations)
        y = torch.stack(sorted_outputs)
        # one hot y
        y = torch.nn.functional.one_hot(y, num_classes=10).float()

        self.X = x.to(self.device)
        self.y = y.to(self.device)


def test():
    problem = BucketSort(3, 'cpu')
    print(problem.X)
    print(problem.y)
    problem = EvenPairs(3, 'cpu')
    print(problem.X)
    print(problem.y)
    problem = FirstA(3, 'cpu')
    print(problem.X)
    print(problem.y)
    problem = FirstLastA(3, 'cpu')
    print(problem.X)
    print(problem.y)
    problem = ParityCheck(3, 'cpu')
    print(problem.X)
    print(problem.y)
    problem = CycleNavigation(4, 5, 'cpu')
    print(problem.X)
    print(problem.y)
    problem = CycleNavigation(4, 3, 'cpu')

if __name__ == '__main__':
    test()
