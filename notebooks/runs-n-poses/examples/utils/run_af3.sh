singularity exec \
     --nv \
     --bind ../inputs/af3:/app/af_input \
     --bind ../outputs/af3:/app/af_output \
     --bind /path/to/af3_params:/app/models \
     --bind /path/to/public_databases:/app/public_databases \
     /path/to/singularity/image.sif \
     python run_alphafold.py \
     --json_path=/app/af_input/8c3u__1__1.A__1.C.json \
     --model_dir=/app/models \
     --db_dir=/app/public_databases \
     --output_dir=/app/af_output
