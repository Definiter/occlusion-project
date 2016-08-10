model_type_strs=(prototype)
test_type_strs=(1k_crop_obj 1k_nocrop_obj)

#for model_type_str in "${model_type_strs[@]}"
#do
#   for test_type_str in "${test_type_strs[@]}"
#    do
#        echo $model_type_str
#        echo $test_type_str
        qsub -q taising -v gpu=0,model_type_str=prototype,model_name=0,test_type_str=1k_crop_obj,test_name=0 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=1,model_type_str=prototype,model_name=0,test_type_str=1k_crop_obj,test_name=20 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=2,model_type_str=prototype,model_name=0,test_type_str=1k_crop_obj,test_name=40 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=3,model_type_str=prototype,model_name=0,test_type_str=1k_crop_obj,test_name=60 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=0,model_type_str=prototype,model_name=0,test_type_str=1k_crop_obj,test_name=80 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=1,model_type_str=prototype,model_name=0,test_type_str=1k_crop_obj,test_name=100 -l nodes=compute-1-1 pbs_test.script

        qsub -q taising -v gpu=0,model_type_str=prototype,model_name=0,test_type_str=1k_nocrop_obj,test_name=0 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=1,model_type_str=prototype,model_name=0,test_type_str=1k_nocrop_obj,test_name=20 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=2,model_type_str=prototype,model_name=0,test_type_str=1k_nocrop_obj,test_name=40 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=3,model_type_str=prototype,model_name=0,test_type_str=1k_nocrop_obj,test_name=60 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=2,model_type_str=prototype,model_name=0,test_type_str=1k_nocrop_obj,test_name=80 -l nodes=compute-1-1 pbs_test.script
        qsub -q taising -v gpu=3,model_type_str=prototype,model_name=0,test_type_str=1k_nocrop_obj,test_name=100 -l nodes=compute-1-1 pbs_test.script



        # qsub -q taising -v gpu=0,net_name=0,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=1,net_name=20,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=2,net_name=40,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=3,net_name=60,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=0,net_name=80,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=1,net_name=100,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=2,net_name=all,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script



        # qsub -q taising -v gpu=0,net_name=0,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=1,net_name=10,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=2,net_name=20,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=3,net_name=30,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-1 pbs_test.script
        # qsub -q taising -v gpu=0,net_name=40,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=1,net_name=50,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=2,net_name=60,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=3,net_name=70,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=0,net_name=80,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=1,net_name=90,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=2,net_name=100,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
        # qsub -q taising -v gpu=3,net_name=all,model_type_str=$model_str,test_type_str=$test_str -l nodes=compute-1-3 pbs_test.script
#        echo ---
#    done
#done
