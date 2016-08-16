func_strs=(train test)
type_strs=(1k_nocrop_obj)
names=(0 20 40 60 80 100 all)

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
            echo ---------
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=0   -l nodes=compute-0-10 -N $func_str\_0\_$type_str pbs_lmdb.script
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=20  -l nodes=compute-0-11 -N $func_str\_20\_$type_str pbs_lmdb.script
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=40  -l nodes=compute-0-12 -N $func_str\_40\_$type_str pbs_lmdb.script
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=60  -l nodes=compute-0-13 -N $func_str\_60\_$type_str pbs_lmdb.script
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=80  -l nodes=compute-0-14 -N $func_str\_80\_$type_str pbs_lmdb.script
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=100 -l nodes=compute-0-15 -N $func_str\_100\_$type_str pbs_lmdb.script
            qsub -q default -v func_str=$func_str,type_str=$type_str,name=all -l nodes=compute-0-16 -N $func_str\_all\_$type_str pbs_lmdb.script
        #done
    done
done

