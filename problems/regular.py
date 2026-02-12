import ast
import sys
import zlib
import copy
import json
import yaml
import torch
import random
import timeit
import numpy as np
import seaborn as sns
import networkx as nx
from PIL import Image
from pathlib import Path
from typing import Union
# import pygraphviz as pgv
from itertools import product, groupby
import torch.nn.functional as nn
import torch.nn.utils.rnn as rnn_utils

import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.patches as mpatches
from torch import Tensor

from utils.dataset import load_language


class RandomAutomaton:
    """
        This class represents a random automaton. An automaton is a finite state machine that accepts or rejects a string of symbols.
        The automaton is represented as a directed graph where each node represents a state and each edge represents a transition between states.
        The automaton is randomly generated based on the provided parameters.

        Attributes:
            sigma (list): The alphabet of the automaton, i.e., the set of symbols that the automaton can process.
            Q (list): The set of states in the automaton.
            edge_values (dict): A dictionary mapping each edge (represented as a tuple of states) to a symbol from the alphabet.
            q0 (int): The initial state of the automaton.
            M (numpy.ndarray): The adjacency matrix of the automaton's graph. M[i, j] = 1 if there is an edge from state i to state j, and 0 otherwise.
            F (list): The set of final states of the automaton.
        """

    def __init__(self, sigma: list, num_states: int, ratio_max_path: float = 0.5, ratio_final_states: float = 0.5,
                 ratio_max_connectivity: float = 1.0, final_states: list = None):
        """
        Initializes a new instance of the RandomAutomaton class.
        Creates a random graph that connects all nodes with a maximum of len(sigma) outgoing edges, makes sure that
        every node can reach a final state and assigns random symbols from sigma to the edges.
        Args:
            sigma (list): The alphabet of the automaton.
            num_states (int): The number of states in the automaton.
            ratio_max_path (float): The maximum length of a random walk as a ratio of the number of states. Default is 0.5.
            ratio_final_states (float): The ratio of final states to all states. Default is 0.5.
            ratio_max_connectivity (float): The ratio of outgoing edges to all possible outgoing edges. Default is 1.0.
        """
        self.name = None
        self.sigma = sigma
        self.Q = list(range(num_states))

        self.edge_values = {}
        # ideally, lookup_table is replaced by graph; kept for compatibility
        self.lookup_table = {g: {} for g in self.Q}
        self.graph = np.zeros((num_states, len(sigma)), dtype=np.int32)
        self.q0 = 0
        self.M = self._create_automaton(ratio_max_path, ratio_max_connectivity)
        # dictionary with indices of finale states as keys and 1 as value;
        # instead: numpy array with labels as entries at corresponding indices
        if final_states is None:
            self.F = self._select_final_states(ratio_final_states)
        else:
            self.F = np.asarray(final_states)
        self.zero_is_final_state = False
        self._assign_edge_values()
        self._create_lookup_table()
        self._create_graph()
        print()

    def __eq__(self, other):
        return np.array_equal(self.graph, other.graph) and self.sigma == other.sigma

    def __str__(self):
        # TODO
        raise NotImplementedError

    def __repr__(self):
        # TODO
        raise NotImplementedError

    def onehot_to_string(self, onehot: torch.Tensor):
        return "".join([self.sigma[i.argmax(dim=-1)] for i in onehot if 1 in i])

    def numeric_to_states(self, sample: torch.Tensor):
        filtered = sample[sample >= 0]
        current_state = self.q0
        states = [current_state]
        for s in filtered:
            next_state = self.graph[current_state][s]
            current_state = next_state
            states.append(current_state)
        return states

    def numeric_to_states_batch(self, batch: torch.Tensor):
        states = torch.full((batch.shape[0], batch.shape[1]+1), -1)
        current_nodes = torch.full((batch.size(0),), self.q0, dtype=torch.int32)
        states[:, 0] = current_nodes
        for i in range(0, batch.size(1)):
            valid_mask = batch[:, i] != -1
            if valid_mask.any():
                # where valid_mask is true, take the value from the graph, else take the previous value (-1 as default)
                next_nodes = torch.where(valid_mask,
                                         torch.tensor(self.graph[current_nodes, batch[:, i]]),
                                         torch.full_like(current_nodes, -1))
                # index zero is the initial state
                states[:, i+1] = next_nodes
                current_nodes = next_nodes
        return states

    def onehot_tensor_to_string(self, onehot: torch.Tensor):
        for i in onehot:
            print(self.onehot_to_string(i))

    def string_to_onehot(self, sample: str, padding=None):
        """
        Padding describes the length of the final padded tensor.
        """
        tensor = torch.from_numpy(np.asarray([self.sigma.index(s) for s in sample]))
        # int 64 type
        tensor = tensor.type(torch.int64)
        onehot = nn.one_hot(tensor, num_classes=len(self.sigma))
        if padding:
            onehot = nn.pad(onehot, (0, 0, 0, padding - onehot.size(0)))
        return onehot

    def random_walk(self, length: int, start_node: int = 0):
        """
        Generate a random walk of given length, starting from start_node = 0.
        Args:
            length: length of the final string from language, i.e. length+1 random walk steps
            start_node: always 0.

        Returns:
            path: list of nodes
        """
        path = [start_node]
        current_node = start_node
        for _ in range(length):
            neighbours = np.where(self.M[current_node, :] == 1)[0]
            if len(neighbours) == 0:
                break
            next_node = np.random.choice(neighbours)
            path.append(next_node)
            current_node = next_node
        return path

    def _create_graph(self):
        for (state1, state2), symbol in self.edge_values.items():
            # Ensure compatibility with single symbols and lists of symbols
            symbols = symbol if isinstance(symbol, list) else [symbol]
            for s in symbols:
                self.graph[state1, self.sigma.index(s)] = state2

    def _get_final_state(self, query: Union[str, np.ndarray]) -> int:
        """
        Given a string, return the final state of the automaton after processing the string.
        """
        if isinstance(query, str):
            final_state = self._final_state_from_string(query)
        elif isinstance(query, np.ndarray) or isinstance(query, tuple):
            final_state = self._final_state_from_ints(query)
        else:
            raise ValueError(f"Query must be either a string or a numpy array, but is {query.dtype}.")
        return final_state

    def _final_state_from_string(self, string: str):
        # Initialize the current state as the initial state of the automaton
        current_state = self.q0

        # Process each symbol in the string
        for symbol in string:
            symbol_idx = self.sigma.index(symbol)
            # Find the state associated with the transition for this symbol
            next_state = self.graph[current_state, symbol_idx]

            # If there is no transition for this symbol, the string is not part of the language
            if next_state is None:
                return False

            # Update the current state
            current_state = next_state
        return current_state

    def _final_state_from_ints(self, array: np.ndarray):
        # Initialize the current state as the initial state of the automaton
        current_state = self.q0

        # Process each int in the array
        for symbol_idx in array:
            # Find the state associated with the transition for this symbol
            next_state = self.graph[current_state, symbol_idx]

            # If there is no transition for this symbol, the string is not part of the language
            if next_state is None:
                return False

            # Update the current state
            current_state = next_state
        return current_state

    def is_part(self, string: str) -> bool:
        """
        Checks if a given string is part of the language. Returns False if it contains symbols that are not part of the
        language, if the string cannot be represented by a path in the automaton, or if the path does not end in a final
        state.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string is part of the language, False otherwise.
        """
        # Check if the string contains only symbols that are known to the automaton
        if not all(symbol in self.sigma for symbol in string):
            return False

        final_state = self._get_final_state(string)

        # After processing all symbols in the string, check if the current state is a final state
        return bool(self.F[final_state])

    def get_label(self, string: str) -> int:
        """
        Returns the label for a given string. Works only if specific values are assigned to the final states.

        Args:
            string (str): The string to get the label for.

        Returns:
            int: The label for the string.
        """
        final_state = self._get_final_state(string)
        return self.F[final_state]

    def states_to_symbols(self, nodes):
        """
        Converts a list of nodes (states) into a list of symbols by looking up the symbol for each edge in the path.
        """
        return [self.edge_values[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]

    def symbols_to_states(self, symbols):
        """
        Converts a list of symbols into a list of nodes (states) by looking up the node for each edge in the path.
        """
        current_state = self.q0
        states = [current_state]
        for s in symbols:
            next_state = self.lookup_table[current_state].get(s, None)
            if next_state is None:
                return None
            current_state = next_state
            states.append(current_state)
        return states

    def _assign_edge_values(self):
        """
        Assigns a random symbol from the alphabet to each edge in the automaton, making sure that each node never has
        multiple outgoing edges with the same symbol.
        """
        indices = np.array(list(zip(*np.where(self.M == 1))))
        symbols = []
        for i in range(len(self.Q)):
            outgoing_edges = indices[indices[:, 0] == i]
            symbols += list(np.random.choice(self.sigma, size=len(outgoing_edges), replace=False))
        indices = [tuple(i) for i in indices]
        self.edge_values = dict(zip(indices, symbols))

    def _create_lookup_table(self):
        for (state1, state2), symbol in self.edge_values.items():
            # should not me needed regarding init of lookup_table
            # if state1 not in self.lookup_table:
            #     self.lookup_table[state1] = {}
            self.lookup_table[state1][symbol] = state2

    def create_dfa_strategy(self, savepath=None):
        all_buckets = []
        used_states = [0]
        bucket = [0]
        # create and fill buckets until all states have been put into a bucket
        while len(used_states) < len(self.Q):
            fill_bucket = True
            # fill a single bucket until we created an Öhrchen
            while fill_bucket == True:
                candidates = []
                plan_b_candidates = []
                for state in used_states:
                    for s in self.graph[state]:
                        # good candidates can be reached from the already-used states and are final states
                        if s not in used_states and s in self.F:
                            candidates.append(s)
                        # plan b candidates can be reached from the already-used states, but are not final states
                        elif s not in used_states:
                            plan_b_candidates.append(s)
                if not candidates:
                    candidates = plan_b_candidates
                # create a rating for the candidates,
                # favouring the ones that have the most connections to the already-used states
                # we can use self.M for modular arithmetic, because we are only interested in whether there exists an edge at all
                b_rating = [sum(np.isin(np.where((self.M[c, :] == 1))[0], used_states).astype(int)) for c
                            in candidates]
                # very good candidates create Öhrchen to the already-used states
                very_good_candidates = [c for c, state in product(candidates, used_states) if
                                        self.M[c][state] == 1]

                if very_good_candidates:
                    pick = candidates[np.argmax(b_rating)]
                    bucket.append(int(pick))
                    used_states.append(pick)
                    all_buckets.append(bucket)
                    fill_bucket = False
                    bucket = []
                # there might be none, if so, pick the best one according to their connections in the automaton
                else:
                    pick = candidates[np.argmax(b_rating)]
                    bucket.append(int(pick))
                    used_states.append(pick)
        if savepath:
            # save all buckets as json file
            with open(savepath, "w") as f:
                json.dump(all_buckets, f)
            print(f"Saved dfa strategy to {savepath}")
        else:
            return all_buckets

    def graph_to_nx_graph(self):
        """
        Converts the adjacency matrix of the automaton to a directed graph using the NetworkX library.

        Returns:
            networkx.DiGraph: The directed graph representation of the automaton.
        """
        G = nx.DiGraph()
        for i in range(len(self.M)):
            G.add_node(i, label=i)
            for j in range(len(self.M[i])):
                if self.M[i][j] == 1:
                    G.add_edge(i, j, label=self.edge_values[(i, j)])
        return G

    def plot_nx(self, edge_labels=None, edge_colors=None, edge_widths=None):
        """
        Plots the automaton as a directed graph using the NetworkX library.

        Note:
            This method requires the NetworkX library.
        """
        import networkx as nx

        G = self.graph_to_nx_graph()
        pos = nx.kamada_kawai_layout(G)
        node_labels = nx.get_node_attributes(G, 'label')
        # node colors from colorblind palette and node labels
        cmap = plt.get_cmap('jet')
        node_colors = [cmap(i / len(self.F)) for i in self.F]
        if edge_labels is None:
            edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=18)
        # draw edges with arrows
        if edge_colors is None:
            edge_colors = 'black'
        if edge_widths is None:
            edge_widths = 1.0
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, connectionstyle='arc3,rad=0.2', arrowsize=20,
                               min_source_margin=20, min_target_margin=20,
                               width=edge_widths)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, verticalalignment='top', connectionstyle='arc3,rad=0.2')
        plt.show()

    def plot(self, save=None):
        """
        Plots the automaton as a directed graph.

        Args:
            save (str, optional): If provided, the plot will be saved to this file path. Defaults to False.

        Note:
            This method requires the pygraphviz library. Will save temporary files to /tmp .
        """
        # Create a directed graph from the adjacency matrix
        graph = pgv.AGraph(directed=True)

        colors = sns.color_palette("colorblind", len(set(self.F))).as_hex()

        # Add nodes and edges to the graph
        for i in range(len(self.M)):
            # Start node
            if i == 0:
                graph.add_node(i, fillcolor=colors[self.F[i]], color='black', shape='circle', style='filled',
                               penwidth=1.5)
            else:
                graph.add_node(i, fillcolor=colors[self.F[i]], shape='circle', style='filled', penwidth=0)

            for j in range(len(self.M[i])):
                if self.M[i][j] == 1:
                    graph.add_edge(i, j, label=self.edge_values[(i, j)])

        # Save the graph in dot format
        graph.write('/tmp/graph.dot')

        # Render the graph to a file (e.g., in PNG format)
        graph.draw('/tmp/graph.png', prog='dot', format='png', args='-Gdpi=200')
        image = Image.open('/tmp/graph.png')
        width, height = image.size
        plt.figure(figsize=((width / 100), (height / 100)))
        # Create patches for each node color
        start_patch = mpatches.Patch(facecolor=colors[self.F[0]], edgecolor='black', label=f'Label {self.F[0]} & Start')
        other_patches = [mpatches.Patch(facecolor=colors[i], label=f'Label {i}') for i in list(set(self.F[1:]))]

        # Add patches to the legend
        plt.legend(handles=[start_patch, *other_patches], loc='lower center', bbox_to_anchor=(0.5, -0.15))
        # plt.title(f"{save.split('/')[-2]}")
        # Display the graph
        i = imread('/tmp/graph.png')
        plt.imshow(i)
        plt.axis('off')
        # Display the legend
        if save is not None:
            plt.title(f"Problem {save.split('/')[-2].split('_')[-1]}")
            plt.savefig(save)
            print(f"Saved plot to {save}")
        plt.show()

    def _create_automaton(self, max_path_length, ratio_max_connectivity):
        """
        Creates the automaton as a directed graph.

        Args:
            max_path_length (float): The maximum length of a random walk as a ratio of the number of states.
            ratio_max_connectivity (float): The ratio of outgoing edges to all possible outgoing edges.

        Returns:
            numpy.ndarray: The adjacency matrix of the automaton's graph.
        """
        M = np.zeros((len(self.Q), len(self.Q)))
        visited = set()
        possible = copy.copy(self.Q)
        current = 0
        visited.add(current)

        def _update_status(state):
            if sum(M[state, :]) == int(len(self.sigma) * ratio_max_connectivity) and state in possible:
                possible.remove(state)
            visited.add(state)

        # initialise a graph that connects all nodes
        while len(visited) < len(self.Q):
            # length of random walk
            low = 1
            # round up to the nearest integer
            high = max(int(np.ceil(max_path_length * len(self.Q))), 2)
            rand = np.random.randint(low=low, high=high)
            for _ in range(rand):
                # if current has as many outgoing edges as the size of our alphabet
                if sum(M[current, :]) == int(len(self.sigma) * ratio_max_connectivity):
                    try:
                        current = np.random.choice(list(set(possible) & visited))
                    except ValueError:
                        # shit happens, let's start over
                        return self._create_automaton(max_path_length, ratio_max_connectivity)
                    _update_status(current)
                choices = list(np.where(M[current] == 0)[0])
                # prevent self-loops for now
                if current in choices:
                    choices.remove(current)
                next_state = np.random.choice(choices)
                M[current, next_state] = 1
                _update_status(next_state)
                _update_status(current)
                current = next_state
            if list(visited & set(possible)):
                current = np.random.choice(list(visited & set(possible)))
            else:
                # shit happens, let's start over
                return self._create_automaton(max_path_length, ratio_max_connectivity)

        assert np.all(M.sum(axis=1) <= len(self.sigma)), "Some states have too many outgoing edges"
        assert np.all(M[:, 1:].sum(axis=0) > 0), "Some states are not reachable"

        # complete the graph
        M2 = copy.copy(M)
        # for nodes in M with less than len(sigma) outgoing edges, add random edges that create a cycle
        new_possible = [i for i in range(len(self.Q)) if sum(M2[i, :]) < len(self.sigma)]
        try:
            all_preds = [self.get_all_predecessors_to_root(node, M2) for node in new_possible]
        except RecursionError:
            # should not be needed anymore
            print(f"This one is fucked up {M2}")

        for node, preds in zip(new_possible, all_preds):
            while sum(M2[node, :]) < int(len(self.sigma) * ratio_max_connectivity):
                if preds:
                    # if we have predecessors (left), pick one at random
                    pick = np.random.choice(preds)
                    M2[node, pick] = 1
                    preds.remove(pick)
                else:
                    # else, pick a random node
                    pick = np.random.choice(range(len(self.Q)))
                    M2[node, pick] = 1

        return M2

    def _select_final_states(self, ratio):
        """
        This method first selects the minimum number of final states that are needed to make a final state reachable
        from every state in the graph. If the minimum is less than the wanted number of final states, random
        additional final states are added.
        Args:
            ratio (float): The ratio of final states to all states.

        Returns:
            list: The list of final states of the automaton.
        """
        # initialise all states as potential final states
        candidates = list(range(len(self.Q)))
        selected = np.zeros(len(self.Q))
        # whether to keep that node as final state or not
        # for each state, create a list of all possible successors
        all_successors = [self.get_all_successors(n, self.M) for n in self.Q]

        # remove only states from the list of final states if this removal will not result in an empty list of successors
        while candidates:
            # will return list of indices for the lists with only one entry
            singletons = [i for i in range(len(self.Q)) if len(all_successors[i]) == 1]
            if singletons:
                for s in singletons:
                    if s in candidates:
                        candidates.remove(s)
                        selected[all_successors[s]] = 1
            try:
                pick = np.random.choice(candidates)
                candidates.remove(pick)
                for entry in all_successors:
                    if pick in entry:
                        entry.remove(pick)
            except ValueError:
                break
        if selected.sum() < int(len(self.Q) * ratio):
            missing = int(len(self.Q) * ratio - selected.sum())
            pick = np.random.choice(np.where(selected == 0)[0], size=missing, replace=False)
            # ((remove comment as not applicable?)) if we have not selected any final states, pick one at random
            selected[pick] = 1
        # return np.where(selected == 1)[0]
        final_states = np.zeros(len(self.Q)).astype(np.int32)
        final_states[np.where(selected == 1)[0]] = 1
        return final_states

    def get_all_successors(self, node, M):
        """
        Gets all successors of a node in the automaton.

        Args:
            node (int): The node for which to get the successors.
            M (numpy.ndarray): The adjacency matrix of the automaton's graph.

        Returns:
            list: The list of successors of the node.
        """
        successors = set()
        stack = [node]
        visited = set()

        while stack:
            current = stack.pop()
            visited.add(current)
            for i in np.where(M[current, :] == 1)[0]:
                successors.add(i)
                if i not in visited:
                    stack.append(i)
        return list(successors)

    def get_all_predecessors_to_root(self, state, M):
        """
        Gets all predecessors of a state that lead to the root in the automaton. Recursive implementation of a
        depth-first search.

        Args:
            state (int): The state for which to get the predecessors.
            M (numpy.ndarray): The adjacency matrix of the automaton's graph.

        Returns:
            list: The list of predecessors of the state.
        """
        paths = []

        def dfs_preds(current_node, path):
            # breaks if root or if we have a cycle
            if current_node == 0:
                path.append(current_node)
                paths.extend(path.copy())
            elif current_node in path:
                paths.extend(path.copy())
            else:
                path.append(current_node)
                for node in np.where(M[:, current_node] == 1)[0]:
                    dfs_preds(node, path)
            # path.pop()

        dfs_preds(state, [])
        return paths

    def get_output_labels_optimized(self, inputs) -> np.ndarray:
        """
        Generate labels for the input sequences using an optimized approach.
        Assumes self.F is a numpy array containing the label for each state.
        """
        # Initialize result array with zeros
        result = np.zeros((len(inputs), len(set(self.F))), dtype=np.uint8)

        # Convert inputs to a numpy array for efficient processing
        inputs_array = np.array(inputs)

        # Initialize the current states for all sequences to the initial state
        current_states = np.full((inputs_array.shape[0],), self.q0, dtype=np.int32)

        # Iterate over each position in the sequence
        for i in range(inputs_array.shape[1]):
            symbol_indices = inputs_array[:, i]
            # Update current states based on the transitions defined by the symbols
            current_states = self.graph[current_states, symbol_indices]

        # Use advanced indexing to set the appropriate label for each sequence
        result[np.arange(len(inputs)), self.F[current_states]] = 1
        return result

    def get_output_labels_and_states(self, inputs) -> np.ndarray:
        """
        Generate labels for the input sequences using an optimized approach.
        Assumes self.F is a numpy array containing the label for each state.
        """
        # Initialize result array with zeros
        result = np.zeros((len(inputs), len(set(self.F))), dtype=np.uint8)

        # Convert inputs to a numpy array for efficient processing
        inputs_array = np.array(inputs)

        # Initialize the current states for all sequences to the initial state
        current_states = np.full((inputs_array.shape[0],), self.q0, dtype=np.int32)

        # Iterate over each position in the sequence
        for i in range(inputs_array.shape[1]):
            symbol_indices = inputs_array[:, i]
            # Update current states based on the transitions defined by the symbols
            current_states = self.graph[current_states, symbol_indices]

        # Use advanced indexing to set the appropriate label for each sequence
        result[np.arange(len(inputs)), self.F[current_states]] = 1
        return result

    def get_train_batches(self, num_samples, bucket_info, params, seed, fewer_duplicates: bool = False,
                          return_lengths: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a batch of training examples from the automaton using the specified training strategy. All samples
        are always padded to max input length.
        Args:
            num_samples (int): The number of training samples to generate.
            bucket_info (list): The information required to generate the batch based on the training strategy.
            params (dict): Some general parameters for generating the batch (min/ max input length)
            seed (int): The seed for the random number generators (numpy, random, torch)
            fewer_duplicates (bool): Whether to generate fewer duplicates in the batch (relevant for short sequences)
            return_lengths (bool): Whether to not pad the sequences to maximum length.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The input and output sequences of the generated batch
        """
        if params["training_strategy"] == "no-curr":
            # bucket_info will be empty in case of no-curr
            return self.generate_batch(num_samples, (params["min_input_length"], params["max_input_length"]),
                                       seed, fewer_duplicates=fewer_duplicates, return_lengths=return_lengths)
        elif params["training_strategy"] == "zero-curr":
            return self.generate_random_batch(num_samples, (params["min_input_length"], params["max_input_length"]),
                                              seed, fewer_duplicates=fewer_duplicates, return_lengths=return_lengths)
        elif params["training_strategy"] == "dfa":
            # bucket_info contains list of allowed nodes
            return self.generate_dfa_batch_optimized(num_samples, bucket_info,
                                                             (params["min_input_length"], params["max_input_length"]),
                                                             seed, fewer_duplicates=fewer_duplicates, return_lengths=return_lengths)
        elif params["training_strategy"] in ["curr-compression", "anti-curr-compression"]:
            # bucket_info contains interval of allowed compression rates
            return self.generate_compression_batch(num_samples, bucket_info, (params["min_input_length"], params["max_input_length"]), seed, fewer_duplicates=fewer_duplicates, return_lengths=return_lengths)
        elif params["training_strategy"] in ["curr", "mid", "anti"]:
            # bucket_info contains interval of allowed length of input sequences
            bucket_min, bucket_max = min(bucket_info), max(bucket_info)
            if bucket_max < 7:
                fewer_duplicates = False
            return self.generate_batch(num_samples, (bucket_min, bucket_max, params["max_unseen_length"]), seed,
                                       fewer_duplicates=fewer_duplicates, return_lengths=return_lengths)
        else:
            raise ValueError(f"Invalid strategy: {params['strategy']}")

    def generate_random_batch(self, batch_size: int, min_max: tuple[int, int], seed: int,
                              fewer_duplicates: bool = False, return_lengths: bool = False):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lengths = list(range(min_max[0], min_max[1]+1))
        total_vectors = sum(2**n for n in range(min_max[0], min_max[1]+1))
        probabilities = [(2**n) / total_vectors for n in range(min_max[0], min_max[1]+1)]
        selected_lengths = np.random.choice(lengths, size=batch_size, p=probabilities)

        grouped_lengths = [list(group) for _, group in groupby(sorted(selected_lengths))]
        inputs, outputs = [], []
        for samples in grouped_lengths:
            np_rand = np.random.randint(0, len(self.sigma), (len(samples), samples[0]), dtype=np.uint8)
            outputs.extend(self.get_output_labels_optimized(np_rand))
            # create one-hot encoding from the innermost dimension up to i and set the others to zeros
            torch_rand = nn.one_hot(torch.Tensor(np_rand).long(), num_classes=len(self.sigma)).float()
            inputs.extend(torch_rand)

        inputs = rnn_utils.pad_sequence(inputs, batch_first=True, padding_value=0)
        outputs = np.asarray(outputs)

        if return_lengths:
            return inputs, torch.from_numpy(outputs).float(), torch.tensor(sorted(selected_lengths))
        else:
            # inputs and outputs to float32
            return inputs, torch.tensor(outputs).float(), None


    def generate_dfa_batch(self, batch_size: int, bucket_info: list, min_max: tuple[int, int],
                                   seed: int, fewer_duplicates: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of training examples from the automaton using a dfa strategy, identifying nodes that
        should be used to generate the input sequences. Faster version than the original implementation.
        Args:
            batch_size (int): The number of training examples to generate.
            bucket_info (list): The nodes that are allowed to be used in the input sequences.
            min_max: tuple[int, int]: The minimum and maximum length of the input sequences.
            seed (int): The seed for the random number generator.
            fewer_duplicates (bool): Whether to generate fewer duplicates in the batch (relevant for short sequences)
        """

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lengths = [i for i in range(min_max[0], min_max[1] + 1)]
        inputs = torch.zeros((batch_size, min_max[1], len(self.sigma))).long()
        outputs = np.zeros((batch_size, len(set(self.F))), dtype=np.uint8)

        input_indices = np.arange(batch_size)
        input_indices = self.split_indices(lengths, batch_size, fewer_duplicates=fewer_duplicates)

        mask = np.isin(self.graph, bucket_info)
        allowed_indices = np.array([np.where(mask[i])[0] for i in range(mask.shape[0])], dtype=object)

        for i, idxs in enumerate(input_indices):
            for j in idxs:
                start_node = torch.tensor(0)
                current_node = start_node
                for k in range(0, lengths[i]):
                    next_index = np.random.choice(allowed_indices[current_node])
                    next_node = self.graph[current_node, next_index]
                    inputs[j, k, next_index] = 1
                    current_node = next_node
                outputs[j, self.F[current_node]] = 1

        return inputs.float(), torch.from_numpy(outputs).float()

    def generate_dfa_batch_optimized(self, batch_size: int, allowed_nodes: list, min_max: tuple[int, int],
                                             seed: int, fewer_duplicates: bool = False,
                                             return_lengths: bool = False) -> tuple[Tensor, Tensor, Tensor] | \
                                                                                        tuple[Tensor, Tensor, None]:
        """
        Generates a batch of training examples from the automaton using a dfa strategy, identifying nodes that
        should be used to generate the input sequences. Faster version than the original implementation.
        Args:
            batch_size (int): The number of training examples to generate.
            allowed_nodes (list): The nodes that are allowed to be used in the input sequences.
            min_max: tuple[int, int]: The minimum and maximum length of the input sequences.
            seed (int): The seed for the random number generator.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lengths = torch.arange(min_max[0], min_max[1] + 1, dtype=torch.long)
        inputs = torch.zeros((batch_size, min_max[1], len(self.sigma)), dtype=torch.long)
        outputs = torch.zeros((batch_size, len(set(self.F))), dtype=torch.float)
        input_lengths = []

        input_indices = np.arange(batch_size)
        # input_indices = np.array_split(input_indices, batch_size // len(lengths))
        input_indices = self.split_indices(lengths, batch_size, fewer_duplicates=fewer_duplicates)

        mask = torch.isin(torch.tensor(self.graph), torch.tensor(allowed_nodes))

        # allowed indices given each row in self.graph
        allowed_indices = np.array([np.where(mask[i])[0] for i in range(mask.shape[0])], dtype=object)
        # iterate over each index that will be populated in inputs' first dimension (batch_size)
        for i, idxs in enumerate(input_indices):
            # we always start at state zero
            current_nodes = np.full((idxs.shape[0],), self.q0, dtype=np.int32)
            # iterate over each position in the input sequence (input's second dimension)
            for k in range(lengths[i]):
                next_indices = np.array([np.random.choice(allowed_indices[node]) for node in current_nodes])
                next_nodes = self.graph[current_nodes, next_indices]
                # make entry in the one-hot encoding (input's third dimension)
                inputs[idxs, k, next_indices] = 1
                current_nodes = next_nodes
            outputs[idxs, self.F[current_nodes]] = 1
            input_lengths.extend([lengths[i]] * len(idxs))
        if return_lengths:
            return inputs.float(), outputs, torch.tensor(input_lengths)
        else:
            return inputs.float(), outputs, None

    def generate_compression_batch(self, batch_size: int, compression_threshold: float, min_max: tuple[int, int],
                                   seed: int, fewer_duplicates: bool = False,
                                   return_lengths: bool = False) -> (tuple[Tensor, Tensor, Tensor] |
                                                                     tuple[Tensor, Tensor, None]):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lengths = torch.arange(min_max[0], min_max[1] + 1, dtype=torch.long)
        # inputs (batch_size, max_length, len(sigma))
        inputs = torch.zeros((batch_size, min_max[1], len(self.sigma))).long()
        # outputs: (batch_size, 2)
        outputs = np.zeros((batch_size, len(set(self.F))), dtype=np.uint8)
        # split the input indices into len(lengths) parts
        input_indices = np.arange(batch_size)
        # will sample batch_size / len(lengths) input sequences per length
        # input_indices: (len(lengths), batch_size / len(lengths))
        input_indices = self.split_indices(lengths, batch_size, fewer_duplicates=fewer_duplicates)
        input_lengths = []
        max_attempts = 5
        valid_indices = []
        min_interval = compression_threshold[0]
        max_interval = compression_threshold[1]
        for i, idxs in enumerate(input_indices):
            done = []
            attempts = 0
            while len(done) < len(idxs) and attempts < max_attempts:
                np_rand = np.random.randint(0, len(self.sigma), (len(idxs), lengths[i]), dtype=np.uint8)
                compression_rates = np.array(
                    [sys.getsizeof(zlib.compress(entry)) / sys.getsizeof(entry) for entry in np_rand])
                if min_interval is None:
                    mask = compression_rates <= max_interval
                elif max_interval is None:
                    mask = compression_rates >= min_interval
                else:
                    mask = (compression_rates >= min_interval) & (compression_rates <= max_interval)
                done += list(np_rand[mask])
                attempts += 1
            np_rand = np.array(done)[:len(idxs)]
            if len(np_rand) < len(idxs):
                # print("less samples than expected")
                idxs = idxs[:len(np_rand)]
            try:
                outputs[idxs] = self.get_output_labels_optimized(np_rand)
                # create one-hot encoding from the innermost dimension up to i and set the others to zeros
                torch_rand = nn.one_hot(torch.Tensor(np_rand).long(), num_classes=len(self.sigma))
                # pad dimension x of torch_rand with zeros to max_length
                torch_rand = nn.pad(torch_rand, (0, 0, 0, min_max[1] - lengths[i]))
                inputs[idxs] = torch_rand
            except IndexError:
                # print(f"No samples for length {i+1}")
                continue
            valid_indices += list(idxs)
            input_lengths.extend([lengths[i]] * len(idxs))
        if return_lengths:
            return inputs[valid_indices].float(), torch.from_numpy(outputs[valid_indices]).float(), torch.tensor(input_lengths)
        else:
            # inputs and outputs to float32
            return inputs[valid_indices].float(), torch.from_numpy(outputs[valid_indices]).float(), None

    def split_indices(self, lengths: list, batch_size: int, fewer_duplicates: bool = False):
        # split the input indices into len(lengths) parts
        input_indices = np.arange(batch_size)
        # create fewer duplicates for shorter input lengths
        if min(lengths) <= 7 and fewer_duplicates:
            indices_per_length = [min(batch_size // len(lengths), len(self.sigma) ** l) for l in lengths if l <= 7]
            # create a list of slices; each slice with the length of 2**l
            temp = [input_indices[start:end] for start, end in
                    zip(np.r_[0, np.cumsum(indices_per_length)[:-1]], np.cumsum(indices_per_length))]
            missing = np.array_split(input_indices[sum(indices_per_length):], len(lengths) - len(indices_per_length))
            input_indices = temp + missing
        else:
            input_indices = np.array_split(input_indices, len(lengths))
        return input_indices

    def generate_batch(self, batch_size: int, min_max: tuple[int, int], rng: int,
                       fewer_duplicates: bool = False, return_lengths: bool = False) -> tuple[Tensor, Tensor, Tensor] | \
                                                                                        tuple[Tensor, Tensor, None]:
        """
        Generates a batch of training examples from the automaton. Samples are always padded to max input length.
        Args:
            batch_size (int): The number of training examples to generate.
            min_max: tuple[int, int]: The minimum and maximum length of the input sequences.
            rng (int): The seed for the random number generator.
            less_duplicates (bool): Whether to generate fewer duplicates for input sequences of length
            < 7. Default is False.
            pad (bool): whether to pad the input sequences to the maximum length. Default is False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and output tensors.
        """
        torch.manual_seed(rng)
        np.random.seed(rng)
        random.seed(rng)
        # inputs (batch_size, max_length, len(sigma))
        inputs = []
        # outputs: (batch_size, 2)
        outputs = np.zeros((batch_size, len(set(self.F))), dtype=np.uint8)
        lengths = [i for i in range(min_max[0], min_max[1] + 1)]
        input_indices = self.split_indices(lengths, batch_size, fewer_duplicates=fewer_duplicates)
        input_lengths = []
        for i, idxs in enumerate(input_indices):
            # in inputs[idxs] set randomly chosen dimensions to 1, ..., len(self.sigma) from 0 to i
            # np_rand: (len(idxs), lengths[i])
            np_rand = np.random.randint(0, len(self.sigma), (len(idxs), lengths[i]), dtype=np.uint8)
            outputs[idxs] = self.get_output_labels_optimized(np_rand)
            # create one-hot encoding from the innermost dimension up to i and set the others to zeros
            # torch_rand = nn.one_hot(torch.Tensor(np_rand).long(), num_classes=len(self.sigma)).float()
            torch_rand = torch.from_numpy(np_rand).type(torch.int8)
            inputs.extend(torch_rand)
            input_lengths.extend([lengths[i]] * len(idxs))

        inputs = rnn_utils.pad_sequence(inputs, batch_first=True, padding_value=-1)

        if return_lengths:
            return inputs, torch.from_numpy(outputs).float(), torch.tensor(input_lengths)
        else:
            # inputs and outputs to float32
            return inputs, torch.from_numpy(outputs).float(), None


class FiniteAutomaton(RandomAutomaton):
    """
        This class represents a finite automaton. An automaton is a finite state machine that accepts or rejects a string of symbols.
        The automaton is represented as a directed graph where each node represents a state and each edge represents a transition between states.
        The automaton is randomly generated based on the provided parameters.

        Attributes:
            sigma (list): The alphabet of the automaton, i.e., the set of symbols that the automaton can process.
            Q (list): The set of states in the automaton.
            edge_values (dict): A dictionary mapping each edge (represented as a tuple of states) to a symbol from the alphabet.
            graph (torch.Tensor): The adjacency matrix of the automaton's graph. graph[i, j] = 1 if there is an edge from state i to state j, and 0 otherwise.
            q0 (int): The initial state of the automaton.
            M (numpy.ndarray): The adjacency matrix of the automaton's graph. M[i, j] = 1 if there is an edge from state i to state j, and 0 otherwise.
            F (list): The set of final states of the automaton.
        """

    def __init__(self, sigma: list, num_states: int, final_states: list, edges: dict[tuple[int, int], str],
                 zero_is_final_state: bool = False):
        """
        Initializes a new instance of the RandomAutomaton class.
        Creates a random graph that connects all nodes with a maximum of len(sigma) outgoing edges, makes sure that
        every node can reach a final state and assigns random symbols from sigma to the edges.
        Args:
            sigma (list): The alphabet of the automaton.
            num_states (int): The number of states in the automaton.
        """
        self.sigma = sigma
        self.Q = list(range(num_states))

        self.edge_values = edges
        self.graph = np.zeros((num_states, len(sigma)), dtype=np.int32)
        self.q0 = 0
        self.M = np.zeros((num_states, num_states))
        self.F = np.asarray(final_states)
        self.zero_is_final_state = zero_is_final_state
        self._create_graph()
        self._create_adjacency_matrix()

    @classmethod
    def from_yaml(cls, path_to_file: str, zero_is_final_state: bool = False):
        absolute_path = Path(__file__).resolve().parent
        with open(absolute_path / "automata" / path_to_file, "r") as f:
            automaton_info = yaml.safe_load(f)
        edge_values = {ast.literal_eval(k): v for k, v in automaton_info["edge_values"].items()}
        # Flatten and handle both single values and lists of symbols in edge_values
        sigma =  sorted(set(
            symbol
            for value in edge_values.values()
            for symbol in (value if isinstance(value, list) else [value])
        ))

        Q = list(set([element for key in edge_values.keys() for element in key]))
        F = automaton_info["final_states"]
        assert len(F) == len(Q), "The number of final states must be equal to the number of states."
        return cls(sigma, len(Q), F, edge_values, zero_is_final_state=zero_is_final_state)

    def get_output_labels(self, inputs) -> torch.Tensor:
        """
        Generate labels for the input sequences.
        """
        result = np.zeros((len(inputs), len(set(self.F))), dtype=np.uint8)
        # iterate over inputs
        for i, sequence in enumerate(inputs):
            current_state = self.q0
            # iterate over the sequence
            for entry in sequence:
                next_state = self.graph[current_state, entry]
                if next_state is None:
                    break
                current_state = next_state
            state_label = self.F[current_state]
            # onehot encoding of the state label
            result[i, state_label] = 1
        return result

    def _create_adjacency_matrix(self):
        for edge, symbol in self.edge_values.items():
            self.M[edge[0], edge[1]] = 1
