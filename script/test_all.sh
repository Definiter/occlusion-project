model_type_strs=(crop_obj)
model_names=(all)
test_type_strs=(crop aperture)
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
            qsub -q taising -v gpu=0,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=0 -l nodes=compute-1-1  -N  ${model_type_str}_${model_name}_${test_type_str}_0_test pbs_test.script
            qsub -q taising -v gpu=1,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=10 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_10_test pbs_test.script
            qsub -q taising -v gpu=2,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=20 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_20_test pbs_test.script
            qsub -q taising -v gpu=3,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=30 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_30_test pbs_test.script
            qsub -q taising -v gpu=0,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=40 -l nodes=compute-1-3  -N ${model_type_str}_${model_name}_${test_type_str}_40_test pbs_test.script
            qsub -q taising -v gpu=1,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=50 -l nodes=compute-1-3  -N ${model_type_str}_${model_name}_${test_type_str}_50_test pbs_test.script
            qsub -q taising -v gpu=2,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=60 -l nodes=compute-1-3  -N ${model_type_str}_${model_name}_${test_type_str}_60_test pbs_test.script
            qsub -q taising -v gpu=3,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=70 -l nodes=compute-1-3  -N ${model_type_str}_${model_name}_${test_type_str}_70_test pbs_test.script
            qsub -q taising -v gpu=0,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=80 -l nodes=compute-1-1  -N ${model_type_str}_${model_name}_${test_type_str}_80_test pbs_test.script
            qsub -q taising -v gpu=1,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=90 -l nodes=compute-1-3  -N ${model_type_str}_${model_name}_${test_type_str}_90_test pbs_test.script
            qsub -q taising -v gpu=0,model_type_str=${model_type_str},model_name=${model_name},test_type_str=${test_type_str},test_name=100 -l nodes=compute-1-3  -N ${model_type_str}_${model_name}_${test_type_str}_100_test pbs_test.script
            echo ---
        done
    done
done
