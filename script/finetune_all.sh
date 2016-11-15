model_type_strs=(every_grs)

for model_type_str in "${model_type_strs[@]}"
do
    echo $model_type_str
    qsub -q taising -v gpu=2,model_name=all,model_type_str=$model_type_str -l nodes=compute-1-1 -N ${model_type_str}_all_finetune pbs_finetune.script 
done

