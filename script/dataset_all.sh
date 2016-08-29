type_strs=(crop_obj)

for type_str in "${type_strs[@]}"
do
    echo $type_str
    qsub -q default -v type_str=${type_str},dataset_index=0 -l nodes=compute-0-10 -N ${type_str}_0_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=1 -l nodes=compute-0-11 -N ${type_str}_1_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=2 -l nodes=compute-0-12 -N ${type_str}_2_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=3 -l nodes=compute-0-13 -N ${type_str}_3_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=4 -l nodes=compute-0-14 -N ${type_str}_4_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=5 -l nodes=compute-0-15 -N ${type_str}_5_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=6 -l nodes=compute-0-16 -N ${type_str}_6_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=7 -l nodes=compute-0-10 -N ${type_str}_7_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=8 -l nodes=compute-0-11 -N ${type_str}_8_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=9 -l nodes=compute-0-12 -N ${type_str}_9_dataset pbs_dataset.script
    sleep 20
    qsub -q default -v type_str=${type_str},dataset_index=10 -l nodes=compute-0-13 -N ${type_str}_10_dataset pbs_dataset.script
    sleep 20
done
