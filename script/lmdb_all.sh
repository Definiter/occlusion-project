func_strs=(test)
type_strs=(crop nocrop)
names=(0 10 20 30 40 50 60 70 80 90 100)

for func_str in "${func_strs[@]}"
do
    for type_str in "${type_strs[@]}"
    do
        #for name in "${names[@]}"
        #do
            echo ---------
            echo $func_str
            echo $type_str
            #echo $name
            qsub -q default -v func_str=${func_str},type_str=${type_str},name=10 -l nodes=compute-0-10 -N ${func_str}_${type_str}_10_lmdb pbs_lmdb.script
            sleep 30
            qsub -q default -v func_str=${func_str},type_str=${type_str},name=30 -l nodes=compute-0-11 -N ${func_str}_${type_str}_30_lmdb pbs_lmdb.script
            sleep 30
            qsub -q default -v func_str=${func_str},type_str=${type_str},name=50 -l nodes=compute-0-12 -N ${func_str}_${type_str}_50_lmdb pbs_lmdb.script
            sleep 30
            qsub -q default -v func_str=${func_str},type_str=${type_str},name=70 -l nodes=compute-0-13 -N ${func_str}_${type_str}_70_lmdb pbs_lmdb.script
            sleep 30
            qsub -q default -v func_str=${func_str},type_str=${type_str},name=90 -l nodes=compute-0-14 -N ${func_str}_${type_str}_90_lmdb pbs_lmdb.script
            sleep 30
            echo ---------
        #done
    done
done
