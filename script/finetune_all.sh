qsub -q taising -v gpu=0,net_name=0,crop_str=nocrop   -l nodes=compute-1-1 pbs_finetune.script 
qsub -q taising -v gpu=1,net_name=25,crop_str=nocrop  -l nodes=compute-1-1 pbs_finetune.script 
qsub -q taising -v gpu=2,net_name=33,crop_str=nocrop  -l nodes=compute-1-1 pbs_finetune.script 
qsub -q taising -v gpu=3,net_name=50,crop_str=nocrop  -l nodes=compute-1-1 pbs_finetune.script 
qsub -q taising -v gpu=0,net_name=66,crop_str=nocrop  -l nodes=compute-1-3 pbs_finetune.script 
qsub -q taising -v gpu=1,net_name=80,crop_str=nocrop  -l nodes=compute-1-3 pbs_finetune.script 
qsub -q taising -v gpu=2,net_name=all,crop_str=nocrop -l nodes=compute-1-3 pbs_finetune.script 
