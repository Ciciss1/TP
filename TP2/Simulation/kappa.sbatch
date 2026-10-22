#!/bin/bash
#SBATCH --job-name=kappa
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

# Calibration of KAPPA: 4 + 4 sizes x 6 shapes x 9 angles x 2 constructions = 436 AIREBO relaxations,
# one per core (~1 min for n = 19 up to ~10 min for n = 45). Submit from TP2/Simulation; if the job
# is stopped, submitting it again only runs the relaxations missing from kappa.jsonl.

module purge
module load gcc/13.2.0 python
source ~/venvs/tp/bin/activate
export PATH=$HOME/opt/lammps/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

OUT=/scratch/$USER/kappa
mkdir -p $OUT
cd $SLURM_SUBMIT_DIR

python calibrate_kappa.py run $OUT/kappa.jsonl --n 19,26,32,45 --theta 2,4,6,8,11,14,17,20,24 \
       --seeds 6 --jitter 0.15 --core 0.25 --workers $SLURM_CPUS_PER_TASK
python calibrate_kappa.py analyse $OUT/kappa.jsonl --core 0.25 --fig $OUT/kappa.png | tee $OUT/kappa_summary.txt
