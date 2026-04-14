models=$(find "../outputs/af3/" -type f -name "model.cif")

for line in $models; do
    target_id=$(echo "$line" | cut -d'/' -f4)
    seed=$(echo "$line" | cut -d'/' -f5)
    pdb_id=${target_id:0:4}
    desc=${target_id:4}
    desc_upper=$(echo "$desc" | tr '[:lower:]' '[:upper:]')
                        
    target="$pdb_id$desc_upper"

    sdf_files=("../ground_truth/$target/ligand_files/"*.sdf)

    ost compare-ligand-structures \
        -m "$line" \
        -rl "${sdf_files[@]}" \
        -r "../ground_truth/$target/receptor.cif" \
        -o "../analysis/af3/"$target"_"$seed".json" \
        --lddt-pli --rmsd --lddt-pli-amc
done