model_type_strs=(crop)
model_names=(80)
test_type_strs=(crop nocrop)
#test_names=(0 20 40 60 80 100)


for model_type_str in "${model_type_strs[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for test_type_str in "${test_type_strs[@]}"
        do
            echo $model_type_str
            echo $model_name
            echo $test_type_str
            #qsub -q taising -v gpu=0,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=0 -l nodes=compute-1-1   -N ${model_type_str}_${model_name}_${test_type_str}_0 pbs_test.script
            #sleep 30
            #qsub -q taising -v gpu=1,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=20 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_20 pbs_test.script
            #sleep 30
            qsub -q taising -v gpu=2,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=40 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_40 pbs_test.script
            sleep 30
            #qsub -q taising -v gpu=3,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=60 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_60 pbs_test.script
            #sleep 30
            qsub -q taising -v gpu=0,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=80 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_80 pbs_test.script
            #sleep 30
            #qsub -q taising -v gpu=1,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=100 -l nodes=compute-1-1 -N ${model_type_str}_${model_name}_${test_type_str}_100  pbs_test.script
            #sleep 30
            echo ---
        done
    done
done
