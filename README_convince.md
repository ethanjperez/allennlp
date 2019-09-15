# Finding Generalizable Evidence by Learning to Convince Q&A Models

TODO: Replace below with paper GIF.

<p align="center"><img width="40%" src="doc/static/allennlp-logo-dark.png" /></p>

## Code Overview

Our code was forked from AllenNLP ([Jan 18, 2019 commit](https://github.com/allenai/allennlp/blob/11d8327890bf3665fe687b1284f280a2a3974931)).
Our paper's core code involves changes/additions to AllenNLP in the below files and folders:
<table>
<tr>
    <td><b> allennlp/training/trainer.py </b></td>
    <td> The main training logic for Judge Models and Evidence Agent. </td>
</tr>
<tr>
    <td><b> allennlp/commands/train.py </b></td>
    <td> Command line flags to train BERT Judge Models and Evidence Agents </td>
</tr>
<tr>
    <td><b> allennlp/data/dataset_readers/reading_comprehension/ </b></td>
    <td> Code to read datasets </td>
</tr>
<tr>
    <td><b> allennlp/tests/fixtures/data/ </b></td>
    <td> Mini datasets files for debugging </td>
</tr>
<tr>
    <td><b> datasets/ </b></td>
    <td> Folder for datasets </td>
</tr>
<tr>
    <td><b> eval/ </b></td>
    <td> Evidence Agent sentence selections, which we used for human evaluation (eval/mturk/) and testing for improved Judge generalization (eval/generalization/) </td>
</tr>
<tr>
    <td><b> fasttext/ </b></td>
    <td> Code for training FastText Judge Models and Search-based Evidence Agents </td>
</tr>
<tr>
    <td><b> scripts/ </b></td>
    <td> TODO </td>
</tr>
<tr>
    <td><b> tf_idf/ </b></td>
    <td> Code for training TF-IDF Judge Models and Search-based Evidence Agents </td>
</tr>
<tr>
    <td><b> training_config/ </b></td>
    <td> Config files for training models with various hyperparameters </td>
</tr>
<tr>
    <td><b> tmp/ </b></td>
    <td> Folder for trained models </td>
</tr>
</table>

TODO: Rename folders in eval/ to match paper

In the code, we refer to the Judge Model as "judge" and Evidence Agents as "debaters".

## Installation

TODO: Fix/verify the below commands
TODO: Add other necessary requirements from your allennlp env.

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP.  If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```bash
    conda create -n allennlp python=3.6
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```bash
    source activate allennlp
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

   ```bash
   pip install allennlp
   ```

## Downloading Data

TODO: Add instructions with appropriate names

From the base directory (convince/allennlp), make a folder to store datasets:

   ```bash
   mkdir datasets
   ```

Download RACE using the Google form linked on [this page](http://www.cs.cmu.edu/~glai1/data/race/).
You'll immediately receive an email with a link to the dataset, which you can download with:

   ```bash
   wget [link] -O datasets/race_raw.tar.gz
   tar -xvzf race_raw.tar.gz
   ```

Download DREAM:

   ```bash
   mkdir datasets/dream
   for SPLIT in train dev test; do
     wget https://github.com/nlpdata/dream/blob/master/data/$SPLIT.json -O datasets/dream/$SPLIT.json
   done
   ```

## Training a BERT Judge Model

The below command gave us a BERT QA model with 66.32% validation accuracy at epoch 5:

   ```bash
   allennlp train training_config/race.best.jsonnet -s tmp/race.best.f --debate_mode f --accumulation_steps 32
   ```

#### Pre-trained Judge Models

TODO: Zip all models into single file and change link.
Download the Judge models from our paper [here](https://drive.google.com/open?id=1vJPhOlIAXpYhRjYNEH0B6tqi2KCEKqRu). 

## Training Evidence Agents



#### Pre-trained Evidence Agents

TODO: Zip all models into single file and change link.
Download the Judge models from our paper [here](https://drive.google.com/open?id=1vJPhOlIAXpYhRjYNEH0B6tqi2KCEKqRu).

## Implementation Notes

- We adapted span-based QA code and models for multiple-choice.

## Citing

If you found our code useful, please consider citing our paper:

```
@inproceedings{perez-etal-2019-finding,
    title = "Finding Generalizable Evidence by Learning to Convince Q\&A Models",
    author = "Perez, Ethan and Karamcheti, Siddharth and Fergus, Rob and Weston, Jason and Kiela, Douwe and Cho, Kyunghyun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
}
```
