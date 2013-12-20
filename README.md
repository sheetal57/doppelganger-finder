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

Algorithm:

Let, A = set of users in a dataset
Calculate probability for every pair of users in A

2.a. For a user Ai in A:

       C <- classifier trained of all users in A except Ai
       R <- Test C using Ai and get the probability scores of Ai's document written by other users in A
       for an author Aj in R:
           p[Ai][Aj] = probability of Ai's document was written by Aj
Calculate combined pairwise probability

3.a For Ai, Aj in A:

      p[Ai][Aj] = probability of Ai's document was written by Aj
      p[Aj][Ai] = probability of Aj's document was written by Ai
      p[Ai==Aj] = Ai and Aj are the same author = p[Ai][Aj] * p[Aj][Ai]
      Ai, Aj are likely to be same person if p[Ai==Aj] is high.
      
Output p[Ai==Aj] for every pair of author Ai and Aj in A.
