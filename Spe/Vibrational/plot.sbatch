#!/bin/bash
#SBATCH --job-name=plot_thermalize
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

module purge
module load gcc/13.2.0 python
source ~/venvs/tp/bin/activate
export PATH=$HOME/opt/lammps/bin:$PATH

cd $SLURM_SUBMIT_DIR
python vdos_by_defect.py /scratch/escarmel/crystal_rho0054_big/eps=0/L=544/rho=0.0054/thermalize /scratch/escarmel/crystal_rho0054_big/eps=0/L=544/rho=0.0054