model_type_strs=(crop_obj)

for model_type_str in "${model_type_strs[@]}"
do
    echo $model_type_str
    #qsub -q taising -v gpu=0,model_name=0,model_type_str=$model_type_str   -l nodes=compute-1-5 -N ${model_type_str}_0_finetune pbs_finetune.script 
    #qsub -q taising -v gpu=1,model_name=all,model_type_str=$model_type_str   -l nodes=compute-1-5 -N ${model_type_str}_all_finetune pbs_finetune.script 
    #qsub -q taising -v gpu=0,model_name=20,model_type_str=$model_type_str   -l nodes=compute-1-1 -N ${model_type_str}_20_finetune pbs_finetune.script 
    #qsub -q taising -v gpu=1,model_name=40,model_type_str=$model_type_str   -l nodes=compute-1-1 -N ${model_type_str}_40_finetune pbs_finetune.script 
    #qsub -q taising -v gpu=2,model_name=60,model_type_str=$model_type_str   -l nodes=compute-1-1 -N ${model_type_str}_60_finetune pbs_finetune.script 
    #qsub -q taising -v gpu=3,model_name=80,model_type_str=$model_type_str   -l nodes=compute-1-1 -N ${model_type_str}_80_finetune pbs_finetune.script 
    #qsub -q taising -v gpu=2,model_name=all,model_type_str=$model_type_str   -l nodes=compute-1-1 -N ${model_type_str}_all_finetune pbs_finetune.script 
done

qsub -q taising -v gpu=0,model_name=all,model_type_str=crop   -l nodes=compute-1-5 -N crop_all_finetune pbs_finetune.script 
qsub -q taising -v gpu=1,model_name=all,model_type_str=nocrop   -l nodes=compute-1-5 -N nocrop_all_finetune pbs_finetune.script 
