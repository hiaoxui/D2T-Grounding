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

## Paper

Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data

    @InProceedings{D18-1411,
      author = 	"Qin, G.
            and Yao, J.
            and Wang, X.
            and Wang, J.
            and Lin, C.",
      title = 	"Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data",
      booktitle = 	"Empirical Methods in Natural Language Processing (EMNLP)",
      year = 	"2018",
    }
