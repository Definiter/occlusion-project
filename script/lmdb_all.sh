func_strs=(train val test)
type_strs=(every)
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
            #qsub -q default -v func_str=${func_str},type_str=${type_str},name=all -l nodes=compute-0-10 -N ${func_str}_${type_str}_all_lmdb pbs_lmdb.script
            echo ---------
        #done
    done
done
qsub -q default -v func_str=train,type_str=every,name=all -l nodes=compute-0-10 -N train_every_all_lmdb pbs_lmdb.script
qsub -q default -v func_str=val,type_str=every,name=all -l nodes=compute-0-11 -N val_every_all_lmdb pbs_lmdb.script
qsub -q default -v func_str=test,type_str=every,name=all -l nodes=compute-0-12 -N test_every_all_lmdb pbs_lmdb.script
