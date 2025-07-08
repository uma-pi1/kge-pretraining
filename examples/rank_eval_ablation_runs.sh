#!/bin/bash

# args
model=${model:-none}
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

# enforce model
if [ $model == "none" ]; 
then
  echo "ERROR: must specify model."
  exit 1;
fi

# enforce device
if [ $device == "none" ]; 
then
  echo "ERROR: must specify device."
  exit 1;
fi

# set ablation folders
folders="_fb15k-237_div_no_in_out_links _fb15k-237_div_no_link_prediction _fb15k-237_div_no_neighborhoods _fb15k-237_div_no_rel_domains _fb15k-237_div_no_rel_domains_in_out_links _fb15k-237_div_no_rel_pred _fb15k-237_div_no_std_training"

# loop over ablation runs
for folder in $folders ;
do
    # get rank eval variant
    suffix=${folder:14}

    # set folder name
    folder="$model$folder"

    # run rank eval
    ./rank_eval"$suffix"_all_checkpoints.sh --folder $folder --device $device --from $from --to $to --split $split --chunk_size $chunk_size
done
