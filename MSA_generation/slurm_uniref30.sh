#!/bin/bash -l
#SBATCH -D ./
#SBATCH -J uniref30
#SBATCH --time=0-04:00:00
#SBATCH --mem=367000
#SBATCH --cpus-per-task=48
#SBATCH --partition=genomics
#SBATCH --qos=genomics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=rp24
#SBATCH --mail-user=james.lingford@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_OUT
#SBATCH --error=logs/%j.err
#SBATCH --output=logs/%j.out

# ---
# #SBATCH --partition=genomicsb
# #SBATCH --qos=genomicsbq
# ---
conda activate /fs04/scratch2/rp24/jamesl2/MMseqs2_stuff/openfold-3/scripts/snakemake_msa/rp24_scratch2/jamesl2/miniconda/conda/envs/of3-aln-env

snakemake -s MSA_Snakefile \
    --cores 48 \
    --configfile ./msa_uniref30.json \
    --nolock \
    --keep-going \
    --latency-wait 10
