type_strs=(1k_nocrop_obj 1k_crop_obj)

for type_str in "${type_strs[@]}"
do
    echo $type_str
    for index in {0..5}
    do
        echo $index
        qsub -q taising -v type_str=$type_str,dataset_index=$index -l nodes=compute-1-1 pbs_dataset.script
    done
done
