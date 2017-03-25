model_type_strs=(crop_image)
model_names=(all)
#test_type_strs=(crop crop_img crop_obj crop_image aperture)
test_type_strs=(crop)

for model_type_str in "${model_type_strs[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for test_type_str in "${test_type_strs[@]}"
        do
            echo $model_type_str
            echo $model_name
            echo $test_type_str
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=0 --nodelist=compute-1-5 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=1 --nodelist=compute-1-5 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=2 --nodelist=compute-1-5 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=3 --nodelist=compute-1-5 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=4 --nodelist=compute-1-7 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=5 --nodelist=compute-1-7 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=6 --nodelist=compute-1-7 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=7 --nodelist=compute-1-7 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=8 --nodelist=compute-1-9 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=9 --nodelist=compute-1-9 slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=10 --nodelist=compute-1-9 slurm_test.script
            echo ---
        done
    done
done
