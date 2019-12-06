#!/bin/sh

#SBATCH --partition=debug-EL7
#SBATCH --time=15:00

# load Anaconda, this will provide Jupyter as well.
module load Anaconda3/5.3.0

export XDG_RUNTIME_DIR=""

# specify here the directory containing your notebooks
JUPYTER_NOTEBOOK_DIR=

# launch Jupyter notebook
srun jupyter notebook --no-browser --ip=$SLURMD_NODENAME --notebook-dir=$JUPYTER_NOTEBOOK_DIR
