# Pfam-Classification
Prediction of function of protein domains, based on the PFam dataset.

Domains are functional sub-parts of proteins; much like images in ImageNet are pre segmented to
contain exactly one object class, this data is presegmented to contain exactly and only one
domain.

The purpose of the dataset is to repose the PFam seed dataset as a multiclass classification
machine learning task.

The task is: given the amino acid sequence of the protein domain, predict which class it belongs
to. There are about 1 million training examples, and 18,000 output classes.

The dataset follows the structure:

Description of fields:

- *sequence*: These are usually the input features to your model. Amino acid sequence for this domain.
There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite
uncommon: X, U, B, O, Z.

- *family_accession*: These are usually the labels for your model. Accession number in form PFxxxxx.y
(Pfam), where xxxxx is the family accession, and y is the version number.
Some values of y are greater than ten, and so 'y' has two digits.
family_id: One word name for family.

- *sequencename*: Sequence name, in the form "$uniprotaccessionid/$startindex-$end_index".
aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of
the family in seed, with gaps retained.

In this project, we develop five different approaches: BLAST (alignment-based method), TFIDF, LSTM-based method, CNN-based method and a protein language model based method (ESM)

To run the project and retrieve the results, please follow this [Colab Notebook](https://colab.research.google.com/drive/126pLNqShG515j9VmEMJohU6SVa2WEvZw?usp=sharing).

You can find the results of the different methods in the results folder.

An deeper explanation of the different approaches will be included in the presentation.