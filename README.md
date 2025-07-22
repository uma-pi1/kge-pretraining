# KGE Pre-Training

This is the repository for the paper 
[Beyond Link Prediction: On Pre-Training Knowledge Graph Embeddings](https://aclanthology.org/2024.repl4nlp-1.11/) published at Rep4NLP@ACL2024. 
In this paper, we encourage the study of new ways to train knowledge graph
embedding models, so they may better capture a knowledge graph, and thus
potentially be more useful at downstream tasks. 

## Installation

This codebase is based on [LibKGE](https://github.com/uma-pi1/kge), so the 
installation process is similar:

```sh
git clone https://github.com/uma-pi1/kge-pretraining.git
cd kge-pretraining
pip install -e .
```

Note that many dependencies are given with a specific version number in 
[setup.py](setup.py), and this code may not work if those versions are not
correctly installed.

## Training Models

The [examples](examples/) folder includes some examples of configuration files 
used to train models, including hyperparameter optimization. For example, here's 
how to train a DistMult model on all training objectives except the two tasks 
for link prediction:

```sh
cd examples/distmult_fb15k-237_no_link_prediction
kge resume . --job.device <your_device_id>
```

Each trial from the hyperparameter optimization process will create a different
training job, whose folder is one of the many resulting numbered subfolders
inside the job folder were we run the command to start training.
Each training job folder contains all checkpoints from that job.

To train models using different combinations of the pre-training objectives 
studied in the paper, create new configuration files following those found in 
the [examples](examples/) folder. 
For pre-training on Wikidata5M, please get the training data from LibKGE.

## How To Evaluate Models

The [examples](examples/) folder also includes several scripts that should make 
it easy to evaluate all checkpoints both on the pre-training tasks as well as on 
downstream tasks.
For example, here's how to get the performance on all pre-training tasks of all 
checkpoints that resulted from the job we ran in the example above for training 
models:

```sh
cd examples
./rank_eval_all_checkpoints.sh --folder distmult_fb15k-237_no_link_prediction --device <your_device_id>
```

Similarly, here's how you can evaluate those same checkpoints on the entity 
classification tasks for the FB15K-237 dataset:

```sh
cd examples
./237_ec_all_checkpoints.sh --folder distmult_fb15k-237_no_link_prediction --device <your_device_id>
```

The [examples](examples/) folder contains various other scripts for evaluating 
on other combinations of pre-training tasks or on the downstream tasks for the 
YAGO3-10 and Wikidata5M datasets.
To evaluate on downstream tasks from Wikidata5M, you first need to uncompress
the training data of each task.

## How to Add New Training Objectives

The main extensions to LibKGE for training models are found in the following 
files:

* [train_1vsAll_hybrid.py](kge/job/train_1vsAll_hybrid.py): 1vsAll generalized for more tasks
* [train_KvsAll_hybrid.py](kge/job/train_KvsAll_hybrid.py): KvsAll generalized for more tasks
* [train_1vsAll_hybrid.py](kge/job/train_negative_sampling_hybrid.py): NegSamp generalized for more tasks

Note that they all follow the internal notation of representing task queries 
using an underscore to mark the target of query, and the symbol *^* to mark 
a wildcard slot.
E.g. *s^_* represents the (subject) entity neighborhood task. 
To add a new training objective, we suggest adding a new training file 
following any of the files listed above as an example. Similarly, to evaluate a 
new task you implement for a training objective, you should add support for that
task in the [RankingEvaluation](kge/job/eval_ranking.py) job, where we 
implemented the more general evaluation protocol that includes all tasks 
included in our study.

## How to Add New Downstream Tasks

Each downstream task is associated to an existing pre-training graph, so you
would first have to add that data to the corresponding graph, e.g. as done with
[date_of_birth](data/fb15k-237/date_of_birth/) for [FB15K-237](data/fb15k-237/).
The main extension to LibKGE for evaluating on downstream tasks is found in
the file [downstream_task.py](kge/job/downstream_task.py). 
So, in addition to adding your new data, you would have to modify that file to 
support your new downstream tasks.

## How to Cite

If you use our data, code or compare against our results, please cite the 
following publication:

```
@inproceedings{ruffinelli2024beyond,
  title={Beyond Link Prediction: On Pre-Training Knowledge Graph Embeddings},
  author={Ruffinelli, Daniel and Gemulla, Rainer},
  booktitle={Proceedings of the 9th Workshop on Representation Learning for NLP (RepL4NLP-2024)},
  year={2024}
}
