"""Run on the cluster to regenerate deidentify_nun.job cleanly."""

job = """\
#!/bin/sh
#SBATCH --job-name=deiden_nun
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --qos=gpu40g
#SBATCH --output=/data/home/sermkiatl/deiden/deiden_%j.out
#SBATCH --error=/data/home/sermkiatl/deiden/deiden_%j.err

DATADIR=/data/home/sermkiatl/deiden

export SINGULARITY_BIND="/data/scratch,/data/home"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

echo "=========================================="
echo "=== Job started : $(date) ==="
echo "=== Node        : $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd ${DATADIR}

singularity exec --nv /data/container_image/pytorch-22.03-py3.sif \\
    pip install -q --user transformers sentencepiece pandas openpyxl tqdm huggingface_hub

singularity exec --nv /data/container_image/pytorch-22.03-py3.sif \\
    python3 deidentify_nun.py

echo "=== Job finished: $(date) - exit $? ==="
"""

with open("deidentify_nun.job", "w") as f:
    f.write(job)

print("Written: deidentify_nun.job")
