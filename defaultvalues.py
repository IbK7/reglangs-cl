import os
from pathlib import Path

# export in COMPUTERNAME in your bashrc
computer_name = os.getenv('COMPUTERNAME')

if computer_name == 'marvin':
    PROJECT_PATH = Path("/lustre/scratch/data/vtoborek_hpc-infcurr/reglangs-cl")
    DATASET_PATH = Path("/lustre/scratch/data/vtoborek_hpc-infcurr/data")
    RESULT_PATH = Path('/lustre/scratch/data/vtoborek_hpc-infcurr/results')
elif computer_name == "mlai":
    PROJECT_PATH = Path("/home/mlai51/toborek/projects/reglangs-cl")
    DATASET_PATH = Path("/home/mlai21/share/data/reglangs-cl")
    RESULT_PATH = Path('/home/mlai51/toborek/results/reglangs-cl')
elif computer_name == "mlai_florian":
    PROJECT_PATH = Path("/home/mlai21/share/code/reglangs-cl")
    DATASET_PATH = Path("/home/mlai21/share/data/reglangs-cl")
    RESULT_PATH = Path('/home/mlai21/share/code/reglangs-cl/results')
elif computer_name == "florian":
    PROJECT_PATH = Path("/home/florian/Documents/Code/reglangs-cl")
    DATASET_PATH = Path("/home/florian/Documents/Code/reglangs-cl/data")
    RESULT_PATH = Path('/home/florian/Documents/Code/reglangs-cl/results')
elif computer_name == "iailab73":
    PROJECT_PATH = Path("/home/iailab73/khanm2/reglangs-cl")
    DATASET_PATH = Path("/home/iailab73/khanm2/reglangs-cl/data")
    RESULT_PATH = Path('/home/iailab73/khanm2/reglangs-cl/results')
else:
    raise ValueError("Computer name not recognized.")

