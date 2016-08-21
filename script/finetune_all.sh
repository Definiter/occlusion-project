model_type_strs=(nocrop)

for model_type_str in "${model_type_strs[@]}"
do
    echo $model_type_str
    qsub -q taising -v gpu=1,model_name=100,model_type_str=$model_type_str   -l nodes=compute-1-1 -N ${model_type_str}_100_finetune pbs_finetune.script 

    # qsub -q taising -v gpu=0,model_name=0,model_type_str=$model_type_str   -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=1,model_name=10,model_type_str=$model_type_str  -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=2,model_name=20,model_type_str=$model_type_str  -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=3,model_name=30,model_type_str=$model_type_str  -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=0,model_name=40,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=1,model_name=50,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=2,model_name=60,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=3,model_name=70,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=0,model_name=80,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=1,model_name=90,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=2,model_name=all,model_type_str=$model_type_str  -l nodes=compute-1-3 pbs_finetune.script 
done
