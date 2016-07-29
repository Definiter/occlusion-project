model_arg=nocrop
test_arg=nocrop
echo $model_arg
echo $test_arg
qsub -q taising -v gpu=3,net_name=0,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-1 pbs_test.script
qsub -q taising -v gpu=1,net_name=25,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-1 pbs_test.script
qsub -q taising -v gpu=2,net_name=33,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-1 pbs_test.script
qsub -q taising -v gpu=0,net_name=50,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-3 pbs_test.script
qsub -q taising -v gpu=1,net_name=66,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-3 pbs_test.script
qsub -q taising -v gpu=2,net_name=80,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-3 pbs_test.script
qsub -q taising -v gpu=3,net_name=all,model_crop_str=$model_arg,test_crop_str=$test_arg -l nodes=compute-1-3 pbs_test.script
