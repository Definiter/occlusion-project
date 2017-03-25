model_type_strs=(crop_image)

for model_type_str in "${model_type_strs[@]}"
do
    echo $model_type_str
    sbatch --export=gpu=0,model_name=all,model_type_str=$model_type_str --nodelist=compute-1-7 slurm_finetune.script
done

