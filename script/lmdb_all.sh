func_strs=(train test)
type_strs=(1k_crop_obj)
names=(0 20 40 60 80 100 all)

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
            echo ---------
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=$name -N $func_str\_$name\_$type_str pbs_lmdb.script
        done
    done
done

