for i in {1..5}; do
  seed=$(od -An -N4 -t u4 /dev/urandom | tr -d ' ')
  boltz predict \
    --out_dir "../outputs/boltz/8c3u__1__1.A__1.C/$seed" \
    "../inputs/boltz/8c3u__1__1.A__1.C/input.yaml" \
    --recycling_steps 10 \
    --diffusion_samples 5 \
    --seed $seed
done


