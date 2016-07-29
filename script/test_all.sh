qsub -q taising -v gpu=0,net_name=0 -l nodes=compute-1-1 pbs_test.script 
qsub -q taising -v gpu=1,net_name=25 -l nodes=compute-1-1 pbs_test.script 
qsub -q taising -v gpu=2,net_name=33 -l nodes=compute-1-1 pbs_test.script 
qsub -q taising -v gpu=3,net_name=50 -l nodes=compute-1-1 pbs_test.script 
qsub -q taising -v gpu=0,net_name=66 -l nodes=compute-1-3 pbs_test.script 
qsub -q taising -v gpu=1,net_name=80 -l nodes=compute-1-3 pbs_test.script 
qsub -q taising -v gpu=2,net_name=all -l nodes=compute-1-3 pbs_test.script 
