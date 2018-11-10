# D2T_Grounding

## Introduction

Natural language should always be accompanied with a context,
 and language grounding aims at learning the correspondences between world and texts.
 In this work, the world is represented as structured tables, whose schema is given and fixed.
 We extract a set of semantic tags, which could be executed on the tables,
 and tries to align these tags to the text spans.
 
We adopt hidden semi-Markov models to learn the correspondences, 
the details of which could be found in our paper (see the section: Paper).
Since gold annotations might be very expensive to obtain, 
we use forward-backward algorithm to estimate parameters.

This project provides a toolkit for hidden semi-Markov models training and inference,
as well as an example which works on dataset [RotoWire](https://github.com/harvardnlp/boxscore-data),
which is used in our paper.

## Hidden Semi-Markov Model (Semi-HMMs)

Inside the folder "HMMs" are the components of Semi-HMMs: HMMs itself, 
posterior regularization constraints and forward-backward algorithm.

We tried to make Semi-HMMs as flexible as possible. As you initialize an HMM, 
you need to specify:

+ n_state: The size of state set. Note: the NULL is not included here.
+ null_mode: How do you treat the NULL tag? 
If you set it as "without", NULL tag will not be used. 
If you choose "with", NULL tag will be used, and regarded as an ordinary tag.
If you choose "skip", NULL tag will be used, but will be skipped when calculating the transition probabilities.
+ semi: The maximum length of word span. 
Note: This option will largely affect the speed of training and inference.
+ pr: What kind of constraints do you want to add? 
You could pass your own constraints as a list into HMMs.
+ use_pr_in_decode: If false, pr will be only used during training.

For more details of posterior regularization, 
please refer to the comments of PR.py.
To get an example of the usage of Semi-HMMs, please refer to the experiment part of our project.

## Paper

Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data

    @InProceedings{D18-1411,
      author = "Qin, G. and Yao, J. and Wang, X. and Wang, J. and Lin, C.",
      title = "Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data",
      booktitle = "Empirical Methods in Natural Language Processing (EMNLP)",
      year = "2018",
    }

You can also find our paper in this repo.
