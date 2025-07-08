#!/bin/bash

# args
folder=${folder:-none}
device=${device:-none}
eval_on_test=${eval_on_test:-False}
test_data_n_times=${test_data_n_times:-False}

# DON'T USE THESE
from=${from:-none}
to=${to:-none}
n_jobs=${n_jobs:-10}
z_scores=${z_scores:-True}
log=${log:-False}
selection_metric=${selection_metric:-none}
# combine_train_with_valid=${combine_train_with_valid:-True}
# use_description_embeddings=${use_description_embeddings:-False}

# read terminal arguments
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; 
   then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

# set hardcoded stuff for REG in YAGO3-10
task="regression"
task_names="reg_airport_elevation_above_sea_level,reg_asteroid_absolute_magnitude,reg_municipality_of_germany_area,reg_river_length,reg_village_population,reg_album_publication_date,reg_human_date_of_birth,reg_sports_season_start_time"
selection_metric="rse"

# run it without combining train with valid
./downstream_task_all_checkpoints.sh --folder $folder \
                                     --device $device \
                                     --eval_on_test $eval_on_test \
                                     --from $from \
                                     --to $to \
                                     --z_scores $z_scores \
                                     --log $log \
                                     --n_jobs $n_jobs \
                                     --combine_train_with_valid False \
                                     --task $task \
                                     --task_names $task_names \
                                     --selection_metric $selection_metric \
                                     --test_data_n_times $test_data_n_times

# # enforce folder
# if [ $folder == "none" ]; 
# then
#   echo "ERROR: must specify folder."
#   exit 1;
# fi

# # enforce device
# if [ $device == "none" ]; 
# then
#   echo "ERROR: must specify device."
#   exit 1;
# fi

# # process values for from and to
# if [ $from != "none" ]; 
# then
#     # remove leading zeros
#     from=$(echo $from | sed 's/^0*//')
#     # check that it isn't empty
#     if [ -z "$from" ];
#     then
#         from=0
#     fi
# fi
# if [ $to != "none" ]; 
# then
#     # remove leading zeros
#     to=$(echo $to | sed 's/^0*//')
#     # check that it isn't empty
#     if [ -z "$to" ];
#     then
#         to=0
#     fi
# fi

# # set hardcoded stuff for EC in FB15K-237
# task="regression"
# task_names="reg_born_on_date,reg_created_on_date,reg_destroyed_on_date,reg_died_on_date,reg_happened_on_date"
# models="mlp,linear_regression"
# selection_metric="rse"

# # gogogo
# echo "EVALUATING DOWNSTREAM TASK ON ALL CHECKPOINTS/TRIALS..."
# echo -e "FOLDER: "$folder
# echo -e "TASK: "$task
# echo -e "TASK NAMES: "$task_names
# echo -e "NUM TIMES: "$num_times
# echo -e "EVAL_ON_TEST: "$eval_on_test
# echo -e "N JOBS: "$n_jobs

# cd $folder
# for trial in */ ; 
# do
#     # process trial number
#     # remove trailing bar
#     trial_num=${trial::-1}
#     # remove leading zeros
#     trial_num=$(echo $trial_num | sed 's/^0*//')
#     # check that it isn't empty
#     if [ -z "$trial_num" ];
#     then
#         trial_num=0
#     fi

#     # check that trial is within given range
#     if [ $from != "none" ]; 
#     then
#         diff="$(($trial_num - $from))"
#         if [ $diff -lt 0 ];
#         then
#             continue 
#         fi
#     fi
#     if [ $to != "none" ]; 
#     then
#         diff="$(($trial_num - $to))"
#         if [ $diff -gt 0 ];
#         then
#             continue 
#         fi
#     fi

#     echo -e "\tTRIAL: "$trial
#     cd $trial
#     for task_name in ${task_names//,/ } ;
#     do
#         echo -e "\t\tTask name: "$task_name
#         log="False"
#         if [[ $task_name = "node_importance_1" || $task_name = "population_number" ]] ;
#         then
#             log="True"
#         fi
#         for checkpoint in *.pt ; 
#         do
#             echo -e "\t\t\tCHECKPOINT: "$checkpoint

#             if [ "$eval_on_test" = "True" ] ;
#             then
#                 output_file=${task_name}"_"${folder}"_"${trial%?}"_"${checkpoint%???}"_test.txt"
#             else
#                 output_file=${task_name}"_"${folder}"_"${trial%?}"_"${checkpoint%???}".txt"
#             fi
#             if [ -f "$output_file" ]; then
#                 echo "Skipping as following output file already exists: "$output_file
#                 continue
#             fi

#             # evaluate checkpoint
#             kge valid . --eval.type downstream_task --checkpoint $checkpoint \
#                         --downstream_task.type $task \
#                         --downstream_task.dataset $task_name \
#                         --downstream_task.models $models \
#                         --downstream_task.selection_metric $selection_metric \
#                         --downstream_task.eval_on_test $eval_on_test \
#                         --downstream_task.sobol True \
#                         --downstream_task.n_jobs $n_jobs \
#                         --downstream_task.z_scores $z_scores \
#                         --downstream_task.log $log \
#                         --job.device $device > $output_file
#         done        

#         # create CSV summary for each trial in the folder
#         if [ "$eval_on_test" = "True" ] ;
#         then
#             python ../../create_csv_summary.py --folder $folder --trial $trial --eval $task --task_name $task_name --models $models --test_data True
#         else
#             python ../../create_csv_summary.py --folder $folder --trial $trial --eval $task --task_name $task_name --models $models --test_data False
#         fi
#     done
#     cd ..
# done
# cd ..

# echo "DONE!"
