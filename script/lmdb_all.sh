func_strs=(train test val)
type_strs=(every_grs)
names=(all)

for func_str in "${func_strs[@]}"
do
    for type_str in "${type_strs[@]}"
    do
        for name in "${names[@]}"
        do
            echo ---------
            echo $func_str
            echo $type_str
            echo $name
            qsub -q taising -v func_str=${func_str},type_str=${type_str},name=${name} -l nodes=compute-1-1 -N ${func_str}_${type_str}_${name}_lmdb pbs_lmdb.script
            echo ---------
        done
    done
done
