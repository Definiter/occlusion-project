model_type_str=(nocrop crop)
test_type_str=(nocrop crop)
for model_str in "${model_type_str[@]}"
do
    for test_str in "${test_type_str[@]}"
    do
        echo $model_str
        echo $test_str
        qsub -q taising -v gpu=0,net_name=0,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=1,net_name=25,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=2,net_name=33,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=3,net_name=50,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=0,net_name=66,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        qsub -q taising -v gpu=1,net_name=80,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        qsub -q taising -v gpu=2,net_name=90,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        qsub -q taising -v gpu=3,net_name=all,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        echo ---
    done
done
