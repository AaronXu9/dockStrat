models=$(find "../outputs/chai/" -type f -name "pred.model_*.cif")

for line in $models; do
    target_id=$(echo "$line" | cut -d'/' -f4)
    pdb_id=${target_id:0:4}
    desc=${target_id:4}
    desc_upper=$(echo "$desc" | tr '[:lower:]' '[:upper:]')                        
    target="$pdb_id$desc_upper"
    seed=$(echo "$line" | cut -d'/' -f5)
    model_id_cif=$(basename "$line")
    model_id=$(echo "$model_id_cif" | cut -d'_' -f3 | cut -d'.' -f1)

    sdf_files=("../ground_truth/$target/ligand_files/"*.sdf)

    ost compare-ligand-structures \
        -m "$line" \
        -rl "${sdf_files[@]}" \
        -r "../ground_truth/$target/receptor.cif" \
        -o "../analysis/chai/${target}_${seed}_${model_id}.json" \
        --lddt-pli --rmsd --lddt-pli-amc
done