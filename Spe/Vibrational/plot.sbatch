#!/bin/bash
#SBATCH --job-name=plot_thermalize
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

module purge
module load gcc/13.2.0 python
source ~/venvs/tp/bin/activate
export PATH=$HOME/opt/lammps/bin:$PATH

cd $SLURM_SUBMIT_DIR
python plot_vdos.py /scratch/escarmel/crystal_rho00135_big/eps=0/L=544/rho=0.00135/thermalize