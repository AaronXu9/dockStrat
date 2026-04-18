models=$(find "../outputs/protenix/" -type f -name "receptor.cif")

for line in $models; do
    target_id=$(echo "$line" | cut -d'/' -f4)
    seed=$(echo "$line" | cut -d'/' -f5)
    model_id=$(echo "$line" | cut -d'/' -f7)

    sdf_files=("../ground_truth/$target_id/ligand_files/"*.sdf)
    model_sdf_files=("../outputs/protenix/"$target_id"/"$seed"/predictions/"$model_id"/"*.sdf)

    ost compare-ligand-structures \
        -m "$line" \
        -ml "${model_sdf_files[@]}" \
        -rl "${sdf_files[@]}" \
        -r "../ground_truth/$target_id/receptor.cif" \
        -o "../analysis/protenix/${target_id}_${seed}_${model_id}.json" \
        --lddt-pli --rmsd --lddt-pli-amc
done