type_strs=(crop)

for type_str in "${type_strs[@]}"
do
    echo $type_str
    qsub -q default -v type_str=${type_str},dataset_index=0 -l nodes=compute-0-10 -N ${type_str}_0_dataset pbs_dataset.script
    #for index in {0..2}
    #do
    #    echo $index
    #    qsub -q default -v type_str=${type_str},dataset_index=${index} -l nodes=compute-0-10 -N ${type_str}_${index}_dataset pbs_dataset.script
    #done
    #for index in {3..6}
    #do
    #    echo $index
    #    qsub -q default -v type_str=${type_str},dataset_index=${index} -l nodes=compute-0-11 -N ${type_str}_${index}_dataset pbs_dataset.script
    #done
    #for index in {7..10}
    #do
    #    echo $index
    #    qsub -q default -v type_str=${type_str},dataset_index=${index} -l nodes=compute-0-12 -N ${type_str}_${index}_dataset pbs_dataset.script
    #done
done
