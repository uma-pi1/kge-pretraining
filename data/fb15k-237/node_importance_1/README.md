This are the splits used for the node importance task on the FB15K dataset by
the paper Representation Learning on Knowledge Graphs for Node Importance 
Estimation by Huang et al., KDD21.
The TXT files were created by modifying their code to dump the exact splits they
use and then running the script process_nie_splits.py on them (should be on the
phd/tmp folder).
The DEL files where then created by running the script 
prepare_nie_splits_for_libkge (should also be in the phd/tmp folder).
If there a difference in the number of tuples in a split, it's because the RAW
ones correspond to FB15K, not FB15K-237.
