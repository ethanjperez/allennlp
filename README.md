# Finding Generalizable Evidence by Learning to Convince Q&A Models

TODO: Replace below with paper GIF.

<p align="center"><img width="40%" src="doc/static/allennlp-logo-dark.png" /></p>

## Code Overview

Our code was forked from AllenNLP ([Jan 18, 2019 commit](https://github.com/allenai/allennlp/tree/11d8327890bf3665fe687b1284f280a2a3974931)).
Our paper's core code involves changes/additions to AllenNLP in the below files and folders:
<table>
<tr>
    <td><b> allennlp/training/trainer.py </b></td>
    <td> The main training logic for BERT Judge Models and Evidence Agents </td>
</tr>
<tr>
    <td><b> allennlp/commands/train.py </b></td>
    <td> Command line flags and initial setup to train BERT Judge Models and Evidence Agents </td>
</tr>
<tr>
    <td><b> allennlp/data/dataset_readers/ reading_comprehension/{race,dream}_mc.py </b></td>
    <td> Code to read RACE and DREAM datasets </td>
</tr>
<tr>
    <td><b> allennlp/models/ reading_comprehension/bert_mc.py </b></td>
    <td> Code for BERT QA Models </td>
</tr>
<tr>
    <td><b> allennlp/tests/fixtures/data/ </b></td>
    <td> Mini datasets files for debugging </td>
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
</table>

TODO: Rename folders in eval/ to match paper
TODO: Add code links for above

## Key arguments

In the code, we refer to the Judge Model as "judge" and Evidence Agents as "debaters," following [Irving et al. 2018](https://arxiv.org/abs/1805.00899).
All trained models trained with the ```allennlp train``` command use a BERT architecture.
We use the ```--debate-mode``` flag to indicate what answer an evidence agent aims to support (during training or inference).
We represent each turn as a single character:
<table>
<tr>
    <td><b> Search Agent </b></td>
    <td><b> Learned Agent </b></td>
    <td><b> Evidence Found </b></td>
</tr>
<tr>
    <td> Ⅰ </td>
    <td> ⅰ </td>
    <td> For option 1 </td>
</tr>
<tr>
    <td> Ⅱ </td>
    <td> ⅱ </td>
    <td> For option 2 </td>
</tr>
<tr>
    <td> Ⅲ </td>
    <td> ⅲ </td>
    <td> For option 3 </td>
</tr>
<tr>
    <td> Ⅳ </td>
    <td> ⅳ </td>
    <td> For option 4 (RACE-only) </td>
</tr>
<tr>
    <td> E </td>
    <td> e </td>
    <td> For Every answer option per question </td>
</tr>
<tr>
    <td> L </td>
    <td> l </td>
    <td> For one random answer per question ("Lawyer" - worse than "e" which ensures we train with every answer option) </td>
</tr>
<tr>
    <td> W </td>
    <td> w </td>
    <td> For one random Wrong answer per question </td>
</tr>
<tr>
    <td> A </td>
    <td> a </td>
    <td> For the correct answer ("Alice") </td>
</tr>
<tr>
    <td> B </td>
    <td> b </td>
    <td> Against the correct answer ("Bob") </td>
</tr>
<tr>
    <td> N/A </td>
    <td> f </td>
    <td> Trains a Judge Model via supervised learning </td>
</tr>
</table>

Note that "ⅰ/Ⅰ," "ⅱ/Ⅱ," "ⅲ/Ⅲ," and "ⅳ/Ⅳ," are each one *roman numeral character*; when using these options, just copy and paste the appropriate characters rather than typing "i/I", "ii/II," "iii/III", or "iv/IV."
For our final results, we did not use options "l/L," "w/W," "a/A," or "b/B," but they are implemented and may be useful for others.

To have evidence agents take multiple turns, simply use one character per turn, stringing them together with spaces (when turns are sequential) or without spaces (when turns are simultaneous).
For example, ```--debate-mode ⅰⅱ ⅢⅣ``` first will have learned agents supporting options 1 and 2 choose a sentence each (simultaneously) and then will have search agents supporting options 3 and 4 choose a sentence each (simultaneously).

## Installation

TODO: Fix/verify the below commands
TODO: Add other necessary requirements from your allennlp env.

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment (Python 3.6 or 3.7):

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```bash
    conda create -n convince python=3.6
    ```

3.  Activate the Conda environment

    ```bash
    conda activate convince
    ```

#### Installing the library and dependencies

Clone this repo and move to ```convince/allennlp/``` (where all commands should be run from):
   ```bash
   git clone https://github.com/ethanjperez/convince.git
   cd convince/allennlp
   ```

Install dependencies using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

## Downloading Data

TODO: Add instructions with appropriate names

From the base directory (```convince/allennlp/```), make a folder to store datasets:

   ```bash
   mkdir datasets
   ```

Download RACE using the Google form linked on [this page](http://www.cs.cmu.edu/~glai1/data/race/).
You'll immediately receive an email with a link to the dataset, which you can download with:

   ```bash
   wget [link] -O datasets/race_raw.tar.gz
   tar -xvzf datasets/race_raw.tar.gz
   ```

Here are the RACE dataset subsets we used for [short](https://drive.google.com/open?id=1NtHubMpsz9CUy5_0ZMXdoU6jbJ2BHR18) and [long](https://drive.google.com/open?id=1Hjgs6XMWcSh8AAReLFbaaOy0SBHhw2dQ) passages (place these in ```datasets/```). 
You can split RACE into middle (```race_raw_middle```) and high school (```race_raw_high````) subsets via:
   ```bash
   cp -r datasets/race_raw datasets/race_raw_high
   rm -r datasets/race_raw_high/*/middle
   cp -r datasets/race_raw datasets/race_raw_middle
   rm -r datasets/race_raw_middle/*/high
   ```

Download DREAM:

   ```bash
   mkdir datasets/dream
   for SPLIT in train dev test; do
     wget https://github.com/nlpdata/dream/blob/master/data/$SPLIT.json -O datasets/dream/$SPLIT.json
   done
   ```

Here is the long passage DREAM subset we used for [dev](https://drive.google.com/open?id=15c1B0LRv_RMrtmycrYV1T8zK_n0jlkES) and [test](https://drive.google.com/open?id=174l4d_oz5Qjyp0W8zUUK6JRxgdGqDIlf) (place these in ```datasets/dream```).

Download BERT:
   ```bash
   mkdir -p datasets/bert
   cd datasets/bert
   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
   unzip uncased_L-12_H-768_A-12.zip
   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
   unzip uncased_L-24_H-1024_A-16.zip
   cd ../..
   ```

## Training a BERT Judge Model

The below command gave us a BERT Base QA model (available [here](https://drive.google.com/open?id=1ymA_MziGDYonY3Ck6Wbhss7lSD7AtzX0)) with 66.32% dev accuracy at epoch 5:

   ```bash
   allennlp train training_config/race.best.jsonnet --serialization-dir tmp/race.best.f --debate-mode f --accumulation-steps 32
   ```

TODO: Add BERT Large command and model

## Using Search Agents

The below command will load the judge model as part of an evidence agent (with dummy weights). The agent tries each possible sentence to choose a sentence:
   ```bash
   for DM in Ⅰ Ⅱ Ⅲ Ⅳ; do
     allennlp train training_config/race.best.jsonnet --serialization-dir tmp/race.best.f.dm=$DM --judge-filename tmp/race.best.f/model.tar.gz --eval-mode --debate-mode $DM --search-outputs-path tmp/race.best.f.dm=$DM/search_outputs.pkl
   done
   ```

The above command runs inference over every example in RACE's validation set to find the strongest evidence sentence for each answer.
You can also change the evaluation dataset.
For example to evaluate on RACE's test set, just override the default validation data path by adding ```--overrides "{'validation_data_path': 'datasets/race_raw/test'}"```.

To show a more complicated example, here's how you can run round-robin evidence selections with multiple turns (6 per agent):
   ```bash
   for DM in ⅠⅡ ⅠⅢ ⅠⅣ ⅡⅢ ⅡⅣ ⅢⅣ; do
     allennlp train training_config/race.best.jsonnet --serialization-dir tmp/race.best.f.dm=${DM}_${DM}_${DM}_${DM}_${DM}_${DM} --judge-filename tmp/race.best.f/model.tar.gz --eval-mode --debate-mode $DM $DM $DM $DM $DM $DM --search-outputs-path tmp/race.best.f.dm=${DM}_${DM}_${DM}_${DM}_${DM}_${DM}/search_outputs.pkl
   done
   ```

## Training Learning Agents

With the below commands, you can train a learned agent to predict the search-chosen sentence:
   ```bash
   # Learn to predict search-chosen sentence
   # We got 56.8% accuracy at Epoch 6
   allennlp train training_config/race.best.debate.sl.lr=5e-6.jsonnet --judge-filename tmp/race.best.f/model.tar.gz --debate-mode e --search-outputs-path tmp/race.best.f/search_outputs.pkl --accumulation-steps 12 --reward-method sl --serialization-dir tmp/race.e.c=concat.bsz=12.lr=5e-6.m=sl

   # Learn to predict the Judge Model's probability given each sentence
   # We got 55.1% accuracy at predicting the search-chosen sentence at Epoch 5
   allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet --judge-filename tmp/race.best.f/model.tar.gz --debate-mode e --search-outputs-path tmp/race.best.f/search_outputs.pkl --accumulation-steps 12 --reward-method sl-sents --serialization-dir tmp/race.e.c=concat.bsz=12.lr=1e-5.m=sl-sents

   # Learn to predict the Judge Model's change in probability given each sentence
   # We got 54.3% accuracy at predicting the search-chosen sentence at Epoch 4
   allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet --judge-filename tmp/race.best.f/model.tar.gz --debate-mode e --search-outputs-path tmp/race.best.f/search_outputs.pkl --accumulation-steps 12 --reward-method sl-sents --influence --serialization-dir tmp/race.e.c=concat.bsz=12.lr=1e-5.m=sl-sents.i
   ```
   
Training to convergence takes roughly 1 week on a v100 (16GB).
During the first epoch, we run a search agent to find the judge predictions given each sentence.
We then cache the judge predictions to the file specified after ```--search-outputs-path```.
The cached predictions are used throughout the rest of the training (i.e., epochs after the first are faster).
If you've already train a supervised model, you can save time by training other models simply using the cached predictions from training that model (as in the commands above).  

TODO: Add pre-trained supervised learning models.
TODO: Zip all models into single file and change link.
Download the Judge models from our paper [here](https://drive.google.com/open?id=1vJPhOlIAXpYhRjYNEH0B6tqi2KCEKqRu).

## Implementation Notes

- Some parts of the implementation are BERT-specific (i.e., using [CLS], [SEP], or [MASK] tokens)
- 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ' are special unicode chars. Copy and paste to use elsewhere (don't type directly).
- TODO: Debate modes
- TODO: Reward Methods
- TODO: Explain ToM option
- TODO: Explain RL option

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

Remove variables:
- [Optional] Model.output_type
- [Optional] model.answer_type
- [Optional] bidaf.py
- race.py (span-based reader)
