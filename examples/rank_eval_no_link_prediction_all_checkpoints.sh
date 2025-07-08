#!/bin/bash

# args
folder=${folder:-none}
device=${device:-none}
from=${from:-none}
to=${to:-none}
split=${split:-valid}
chunk_size=${chunk_size:-none}

# read terminal arguments
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; 
   then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

# enforce folder
if [ $folder == "none" ]; 
then
  echo "ERROR: must specify folder."
  exit 1;
fi

# enforce device
if [ $device == "none" ]; 
then
  echo "ERROR: must specify device."
  exit 1;
fi

# process values for from and to
if [ $from != "none" ]; 
then
    # remove leading zeros
    from=$(echo $from | sed 's/^0*//')
    # check that it isn't empty
    if [ -z "$from" ];
    then
        from=0
    fi
fi
if [ $to != "none" ]; 
then
    # remove leading zeros
    to=$(echo $to | sed 's/^0*//')
    # check that it isn't empty
    if [ -z "$to" ];
    then
        to=0
    fi
fi

# gogogo
echo "EVALUATING ALL CHECKPOINTS/TRIALS WITH RANKING EVALUATION..."
echo -e "FOLDER: "$folder
echo -e "SPLIT: "$split
cd $folder
for trial in */ ; 
do
    # process trial number
    # remove trailing bar
    trial_num=${trial::-1}
    # remove leading zeros
    trial_num=$(echo $trial_num | sed 's/^0*//')
    # check that it isn't empty
    if [ -z "$trial_num" ];
    then
        trial_num=0
    fi

    # check that trial is within given range
    if [ $from != "none" ]; 
    then
        diff="$(($trial_num - $from))"
        if [ $diff -lt 0 ];
        then
            continue 
        fi
    fi
    if [ $to != "none" ]; 
    then
        diff="$(($trial_num - $to))"
        if [ $diff -gt 0 ];
        then
            continue 
        fi
    fi

    echo -e "\tTRIAL: "$trial
    cd $trial
    for checkpoint in *.pt ; 
    do
        echo -e "\t\tCHECKPOINT: "$checkpoint
        if [ "$split" = "test" ] ;
        then
            output_file="rank_eval_no_link_prediction_"${folder}"_"${trial%?}"_"${checkpoint%???}"_test.txt"
        else
            output_file="rank_eval_no_link_prediction_"${folder}"_"${trial%?}"_"${checkpoint%???}".txt"
        fi
        if [ -f "$output_file" ]; then
            echo "Skipping as following output file already exists: "$output_file
            continue
        fi

        # evaluate checkpoint
        if [ $chunk_size != "none" ]; 
        then
            kge $split . --eval.type ranking_evaluation --checkpoint $checkpoint \
                        --ranking_evaluation.query_types.sp_ 0.0 \
                        --ranking_evaluation.query_types._po 0.0 \
                        --ranking_evaluation.query_types.s_o 0.0 \
                        --ranking_evaluation.query_types.s^_ 1.0 \
                        --ranking_evaluation.query_types._^o 1.0 \
                        --ranking_evaluation.query_types.^p_ 1.0 \
                        --ranking_evaluation.query_types._p^ 1.0 \
                        --ranking_evaluation.query_types.s_^ 1.0 \
                        --ranking_evaluation.query_types.^_o 1.0 \
                        --ranking_evaluation.chunk_size $chunk_size \
                        --job.device $device > $output_file
        else
            kge $split . --eval.type ranking_evaluation --checkpoint $checkpoint \
                        --ranking_evaluation.query_types.sp_ 0.0 \
                        --ranking_evaluation.query_types._po 0.0 \
                        --ranking_evaluation.query_types.s_o 0.0 \
                        --ranking_evaluation.query_types.s^_ 1.0 \
                        --ranking_evaluation.query_types._^o 1.0 \
                        --ranking_evaluation.query_types.^p_ 1.0 \
                        --ranking_evaluation.query_types._p^ 1.0 \
                        --ranking_evaluation.query_types.s_^ 1.0 \
                        --ranking_evaluation.query_types.^_o 1.0 \
                        --job.device $device > $output_file
        fi
    done

    # create CSV summary for each trial in the folder
    if [ "$split" = "test" ] ;
    then
        python ../../create_csv_summary.py --folder $folder --trial $trial --eval rank_eval_no_link_prediction --test_data True --test_data_n_times False
    else
        python ../../create_csv_summary.py --folder $folder --trial $trial --eval rank_eval_no_link_prediction --test_data False --test_data_n_times False
    fi
    cd ..
done
cd ..

echo "DONE!"
