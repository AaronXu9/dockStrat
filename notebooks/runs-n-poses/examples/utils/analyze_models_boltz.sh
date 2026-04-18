models=$(find "../outputs/boltz/" -type f -name "input_model_*.cif")

for line in $models; do
    target_id=$(echo "$line" | cut -d'/' -f4)
    seed=$(echo "$line" | cut -d'/' -f5)
    model_id_cif=$(basename "$line")
    model_id=$(echo "$model_id_cif" | cut -d'_' -f3 | cut -d'.' -f1)

    sdf_files=("../ground_truth/$target_id/ligand_files/"*.sdf)

    ost compare-ligand-structures \
        -m "$line" \
        -rl "${sdf_files[@]}" \
        -r "../ground_truth/$target_id/receptor.cif" \
        -o "../analysis/boltz/${target_id}_${seed}_${model_id}.json" \
        --lddt-pli --rmsd --lddt-pli-amc
done