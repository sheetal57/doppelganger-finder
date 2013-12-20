doppelganger-finder
===================

Doppelganger-finder finds multiple accounts (doppelgangers) of a user. 
It is useful to merge multiple aliases of a user. 
Given a dataset with n users user_1, user_2, ..., user_n, this code will compute what is the probability that user_i and user_j are the same person, for all i, j where i!=j.

Usage:

    python doppelganger_finder.py -i INPUT -o OUTPUT -s MODEL_DIR

where, INPUT is the file with all features extracted (.arff or .svm),

OUTPUT is the .csv file which will contain the probabilities of two users being the same,

MODEL_DIR is the directory where the classifiers will be saved for future use.

Example:

    python doppelganger_finder.py -i ../example/100-test-all.arff -o ../example/result.csv -s ../example/models/
