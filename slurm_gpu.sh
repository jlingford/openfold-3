#!/bin/bash -l
#SBATCH -D ./
#SBATCH -J openfold3
#SBATCH --mem=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --account=rp24
#SBATCH --gres=gpu:H200:1
#SBATCH --partition=bdi
#SBATCH --qos=bdiq
#SBATCH --time=8:00:00
#SBATCH --mail-user=james.lingford@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_OUT
#SBATCH --error=logs/%j.err
#SBATCH --output=logs/%j.out

# ---
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1

# =================================================================================
# load cuda
module purge
module load cuda/12.2.0

# load conda env
conda activate /fs04/scratch2/rp24/jamesl2/MMseqs2_stuff/openfold-3/rp24_scratch2/jamesl2/miniconda/conda/envs/openfold-3

# export library paths
export OPENFOLD_CACHE=$(pwd)/.cache
export TRITON_CACHE_DIR=$(pwd)/.cache
# export TMPDIR=./tmp/
export CUTLASS_PATH=$(
    python - <<'PY'
import cutlass_library, pathlib
print(pathlib.Path(cutlass_library.__file__).resolve().parent.joinpath("source"))
PY
)
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# docs
run_openfold --help >help.txt
run_openfold predict --help >help_predict.txt

# =================================================================================

CKPTPATH=./.cache/of3-p2-155k.pt
RUNNERYML=./runner.yml

# input
INPUTJSON=./input/group4m.json
# INPUTJSON=./examples/example_inference_inputs/query_protein_ligand_multiple.json

# output name after input file stem name
# WARN: check
name=${INPUTJSON##*/}
name=${name%.*}_test

# run_openfold
run_openfold predict \
    --query-json $INPUTJSON \
    --output-dir ./output/${name} \
    --inference-ckpt-path $CKPTPATH \
    --use-msa-server True \
    --runner-yaml $RUNNERYML
