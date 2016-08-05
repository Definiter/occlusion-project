model_type_str=(crop_obj nocrop_obj)

for model_str in "${model_type_str[@]}"
do
    echo $model_str
    qsub -q taising -v gpu=0,net_name=0,type_str=$model_str   -l nodes=compute-1-1 pbs_finetune.script 
    qsub -q taising -v gpu=1,net_name=20,type_str=$model_str  -l nodes=compute-1-1 pbs_finetune.script 
    qsub -q taising -v gpu=2,net_name=40,type_str=$model_str  -l nodes=compute-1-1 pbs_finetune.script 
    qsub -q taising -v gpu=3,net_name=60,type_str=$model_str  -l nodes=compute-1-1 pbs_finetune.script 
    qsub -q taising -v gpu=0,net_name=80,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    qsub -q taising -v gpu=1,net_name=100,type_str=$model_str -l nodes=compute-1-3 pbs_finetune.script 
    qsub -q taising -v gpu=2,net_name=all,type_str=$model_str -l nodes=compute-1-3 pbs_finetune.script 
    

    # qsub -q taising -v gpu=0,net_name=0,type_str=$model_str   -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=1,net_name=10,type_str=$model_str  -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=2,net_name=20,type_str=$model_str  -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=3,net_name=30,type_str=$model_str  -l nodes=compute-1-1 pbs_finetune.script 
    # qsub -q taising -v gpu=0,net_name=40,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=1,net_name=50,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=2,net_name=60,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=3,net_name=70,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=0,net_name=80,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=1,net_name=90,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
    # qsub -q taising -v gpu=2,net_name=all,type_str=$model_str  -l nodes=compute-1-3 pbs_finetune.script 
done
