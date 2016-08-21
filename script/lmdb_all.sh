func_strs=(train val test)
type_strs=(nocrop)
#names=(0 10 20 30 40 50 60 70 80 90 100 all)
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
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=0  -l nodes=compute-0-10 -N $func_str\_0\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=10 -l nodes=compute-0-10 -N $func_str\_10\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=20 -l nodes=compute-0-12 -N $func_str\_20\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=30 -l nodes=compute-0-11 -N $func_str\_30\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=40 -l nodes=compute-0-14 -N $func_str\_40\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=50 -l nodes=compute-0-12 -N $func_str\_50\_$type_str pbs_lmdb.script

            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=60  -l nodes=compute-0-10 -N $func_str\_60\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=70  -l nodes=compute-0-13 -N $func_str\_70\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=80  -l nodes=compute-0-12 -N $func_str\_80\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=90  -l nodes=compute-0-14 -N $func_str\_90\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=100 -l nodes=compute-0-14 -N $func_str\_100\_$type_str pbs_lmdb.script
            #qsub -q default -v func_str=$func_str,type_str=$type_str,name=all -l nodes=compute-0-15 -N $func_str\_all\_$type_str pbs_lmdb.script

        #done
    done
done

qsub -q default -v func_str=train,type_str=nocrop,name=all -l nodes=compute-0-10 -N train_all_nocrop pbs_lmdb.script
qsub -q default -v func_str=test,type_str=nocrop,name=all -l nodes=compute-0-11 -N test_all_nocrop pbs_lmdb.script
qsub -q default -v func_str=val,type_str=nocrop,name=all -l nodes=compute-0-12 -N val_all_nocrop pbs_lmdb.script
