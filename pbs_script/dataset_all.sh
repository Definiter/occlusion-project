type_strs=(crop_image)

for type_str in "${type_strs[@]}"
do
    echo $type_str
    #qsub -v type_str=${type_str},dataset_index=0 -N ${type_str}_0_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=1 -N ${type_str}_1_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=2 -N ${type_str}_2_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=3 -N ${type_str}_3_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=4 -N ${type_str}_4_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=5 -N ${type_str}_5_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=6 -N ${type_str}_6_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=7 -N ${type_str}_7_dataset pbs_dataset.script
    qsub -v type_str=${type_str},dataset_index=8 -N ${type_str}_8_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=9 -N ${type_str}_9_dataset pbs_dataset.script
    #qsub -v type_str=${type_str},dataset_index=10 -N ${type_str}_10_dataset pbs_dataset.script
done
