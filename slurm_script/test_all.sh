model_type_strs=(crop_image)
model_names=(all)
#test_type_strs=(crop crop_img crop_obj crop_image aperture)
test_type_strs=(crop_img)

for model_type_str in "${model_type_strs[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for test_type_str in "${test_type_strs[@]}"
        do
            echo $model_type_str
            echo $model_name
            echo $test_type_str
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=0 --nodelist=compute-1-5  --output=${model_type_str}_${model_name}_${test_type_str}_0_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=10 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_10_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=20 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_20_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=30 --nodelist=compute-1-5 --output=${model_type_str}_${model_name}_${test_type_str}_30_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=40 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_40_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=50 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_50_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=60 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_60_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=70 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_70_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=80 --nodelist=compute-1-9 --output=${model_type_str}_${model_name}_${test_type_str}_80_test.out slurm_test.script
            #sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=90 --nodelist=compute-1-7 --output=${model_type_str}_${model_name}_${test_type_str}_90_test.out slurm_test.script
            sbatch --export=gpu=0,model_type_str=$model_type_str,model_name=$model_name,test_type_str=$test_type_str,test_name=100 --nodelist=compute-1-9 --output=${model_type_str}_${model_name}_${test_type_str}_100_test.out  slurm_test.script
            echo ---
        done
    done
done
