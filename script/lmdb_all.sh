func_strs=(train test)
type_strs=(1k_nocrop_obj)
names=(0 20 40 60 80 100 all)

#func_strs=(train)
#type_strs=(1k_crop_obj)
#names=(20)

#for func_str in "${func_strs[@]}"
#do
    for type_str in "${type_strs[@]}"
    do
        for name in "${names[@]}"
        do
            echo ---------
            #echo $func_str
            echo $type_str
            echo $name
            echo ---------
            qsub -q taising -v func_str=train,type_str=$type_str,name=$name -l nodes=compute-1-1 pbs_lmdb.script
            qsub -q taising -v func_str=test,type_str=$type_str,name=$name -l nodes=compute-1-1 pbs_lmdb.script
        done
    done
#done

