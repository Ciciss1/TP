#!/bin/bash
#SBATCH --job-name=kappa_scan
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Validation of KAPPA on the Monte Carlo crystals: each state rebuilt by the full pipeline for
# several KAPPA, relaxed excess energies compared. Submit from TP2/Simulation; submitting again
# only runs the crystals missing from scan.jsonl.

module purge
module load gcc/13.2.0 python
source ~/venvs/tp/bin/activate
export PATH=$HOME/opt/lammps/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

OUT=/scratch/$USER/kappa
IN=/scratch/escarmel/crystal_rho00135_kappa_test/eps=0/L=120/rho=0.00135
mkdir -p $OUT
cd $SLURM_SUBMIT_DIR

python kappa_scan.py $IN $OUT/scan2.jsonl --kappa 0.8,1.0,1.3,1.6,2,fit --k 1,2,3 --workers 36 --threads 2
python kappa_scan.py $IN $OUT/scan2.jsonl --summary > $OUT/scan_summary.txt