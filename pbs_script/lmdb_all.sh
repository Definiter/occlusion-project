#func_strs=(train test val)
func_strs=(val)
type_strs=(crop_image)
#names=(0 10 20 30 40 50 60 70 80 90 100 all)
names=(60)

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
            qsub -v func_str=${func_str},type_str=${type_str},name=${name} -N ${func_str}_${type_str}_${name}_lmdb pbs_lmdb.script
            echo ---------
        done
    done
done
