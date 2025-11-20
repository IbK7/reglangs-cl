# %%

# Initialize edge values as a dictionary of lists
edge_values = {}

# Populate edge_values for addition (+), subtraction (-), and multiplication (·)
for i in range(5):
    for j in range(5):
        # Addition
        new_state = (i + j) % 5
        if (i, new_state) not in edge_values:
            edge_values[(i, new_state)] = []
        edge_values[(i, new_state)].append(f"+{j}")

        # Subtraction
        new_state = (i - j) % 5
        if (i, new_state) not in edge_values:
            edge_values[(i, new_state)] = []
        edge_values[(i, new_state)].append(f"-{j}")

        # Multiplication
        new_state = (i * j) % 5
        if (i, new_state) not in edge_values:
            edge_values[(i, new_state)] = []
        edge_values[(i, new_state)].append(f"·{j}")

# Define the final states as modulo results
final_states = [0, 1, 2, 3, 4]

print(edge_values)
print(final_states)

# %%
len(set([v for value in edge_values.values() for v in value]))