'''
Created on Dec 19, 2013

@author: sheetal
'''
from __future__ import print_function, division
import sys
import os
import random
import pylab as pl
import numpy as np
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
from sklearn import grid_search
from sklearn.decomposition import PCA, ProbabilisticPCA
from joblib import Parallel, delayed
import argparse

'''
  input_file = file with features extracted
  outfile = csv file where each line has following format
      user1, user2, probability (user1->user2), probability (user2->user1), combined probability, result
  addn = add this number with the feature values, default 10
          in most cases adding 10 to feature values increase differences between the two classes
  isSparse = if the file is in sparse format
  nthread = number of threads (for quad core 4 or 5 should be good)
  clfname = where the classifiers will be saved
  
'''
def findDoppelgangers(input_file, outfile,  addn, isSparse, nthread, modeldir, clfname):
            
            svm_file = ""
            #convert arff to svm format
            if input_file.endswith('.arff'):
            
                svm_file = input_file.replace('.arff', '.svm')
                with open(input_file, 'r') as arff_fp, open(svm_file, 'w') as svm_fp:
                    if isSparse:
                        fields_table = transform_sparse(arff_fp, svm_fp, False, addn)
                    else:
                        fields_table = transform(arff_fp, svm_fp, False, addn)
            
            elif input_file.endswith('.svm'): 
                svm_file = input_file
                with open(input_file, 'r') as arff_fp, open(svm_file, 'w') as svm_fp:
                    if isSparse:
                        fields_table = transform_sparse(arff_fp, svm_fp, True, addn)
                    else:
                        fields_table = transform(arff_fp, svm_fp, True, addn)
            
            #construct a dict so that author-number is the key and author-name is the value
            authors_to_numbers = dict((v,k) for k,v in fields_table.iteritems())
            print ('Total authors: ', len(authors_to_numbers.keys()))
            print ('Authors are: ',authors_to_numbers.values())
           
            #convert svm to sparse matrix
            data = load_svmlight_file(svm_file)
            
            data_new = PCA(n_components=.999).fit_transform(data[0].toarray())#PCA(n_components=1000).fit_transform(data[0].toarray())
            print ('Total features: ', len(data_new[0]))
        
            #get probability for each author pair
            allAuthors = authors_to_numbers.keys()
            clf = LogisticRegression(penalty='l1')
            
            prob_per_author = getProbsGSThread(nthread, clf, data_new, data[1], allAuthors, modeldir, clfname)
            
            total_prob, add_prob, sq_prob = getCombinedProbs(outfile, prob_per_author, allAuthors, authors_to_numbers)
            
            print ('done')
            
            return total_prob, add_prob, sq_prob
    
    
def getProbsGSThread(nthread, clf, data, label, allAuthors, modeldir, saveModel):
            
            lolo = LeaveOneLabelOut(label)
            
            prob_per_author = [[0]*(len(allAuthors)+3) for i in range(len(allAuthors)+3)]
        
            scores = Parallel(n_jobs=nthread, verbose=5)(delayed(getProbsTrainTest)(clf, data, label, train, test, modeldir, saveModel) for train,test in lolo)
        
            #print (scores)
            for train, test in lolo:
                
                anAuthor = int(label[test[0]])
                #print (anAuthor)
                train_data_label = label[train]
                trainAuthors = list(set(train_data_label))
                test_data_label = label[test]
                nTestDoc = len(test_data_label)
                
                for j in range(nTestDoc):
                    for i in range(len(trainAuthors)):
                        prob_per_author[anAuthor][int(trainAuthors[i])]+= scores[anAuthor-1][j][i]
            
                for i in range(len(trainAuthors)):
                    prob_per_author[anAuthor][int(trainAuthors[i])]/=nTestDoc
            return prob_per_author

'''
       Calculate all probabilities
'''
def getProbsTrainTest(clf, data, label, train, test, modeldir, saveModel):
       
        anAuthor = int(label[test[0]])
        
        print ("current author ", anAuthor)
        
        train_data = data[train,:]
        train_data_label = label[train]
    
        #test on anAuthor
        test_data = data[test,:]
        
        #check if we already have a model
        modelFile = modeldir+ str(anAuthor)+"-"+ saveModel
        
        if os.path.exists(modelFile):
            clf = joblib.load(modelFile)
        else:
             #use the following two lines if you want to choose the regularization parameters using grid search
            #parameters = {'C':[1, 10]}
            #clf = grid_search.GridSearchCV(clf, parameters)
            
            #train
            clf.fit(train_data, train_data_label)
            
            #save model
            joblib.dump(clf, modelFile, compress=9)
            print ("model saved: ", modelFile)
            
        #get probabilities
        scores = clf.predict_proba(test_data)
        return scores
    
def getCombinedProbs(outfile, prob_per_author, allAuthors, authors_to_numbers):
        total_prob = {}
        add_prob = {}
        sq_prob = {}
        
        with open(outfile, "w+") as out:
            out.write('Author 1, Author 2, P(1->2), P(2->1),P(1->2)*P(2->1),(P(1->2)+P(2->1))/2, (P(1->2)^2+P(2->1)^2)/2\n')
            for i in range(len(allAuthors)):
                
                    a = int(allAuthors[i])
                    if len(authors_to_numbers[a])==0:
                        continue
                    for j in range(i+1, len(allAuthors)):
                        b = int(allAuthors[j])
                        
                        result = 0
                        
                        total = prob_per_author[a][b]*prob_per_author[b][a]
                        addition = (prob_per_author[a][b]+prob_per_author[b][a])/2
                        sqsum = (prob_per_author[a][b]*prob_per_author[a][b]+prob_per_author[b][a]*prob_per_author[b][a])/2
                        
                        out.write(authors_to_numbers[a]+" ,"+authors_to_numbers[b]+" ,"+str(prob_per_author[a][b])+","+str(prob_per_author[b][a])+","+str(total)+","+str(addition)+","+str(sqsum)+"\n")
                        
                        if total in total_prob.keys():
                            total_prob[total]+=result
                        else:
                            total_prob[total]=result
                            
                        if addition in add_prob.keys():
                            add_prob[addition]+=result
                        else:
                            add_prob[addition]=result
                        if sqsum in sq_prob.keys():
                            sq_prob[sqsum]+=result
                        else:
                            sq_prob[sqsum]=result
                        
            
        out.close()
        
        return total_prob, add_prob, sq_prob
    
def transform_sparse(arff_fp, svm_fp, svm_exist, addn):
            """Transform every training instance of a sparse ARFF file to SVM instances
               and return all the field mappings collected."""
            ARFF_DELIMITER = ','
            SVM_DELIMITER = ' '

            #reader = csv.reader(arff_fp, delimiter=ARFF_DELIMITER)
            category_table = {}
            counter = 0
            
            if svm_exist:
                for line in svm_fp:
                    fields = line.strip().split(SVM_DELIMITER)
                    
                    if len(fields)<2:
                        continue
                    category = fields[0]
                    
                    rest = fields[:-1]
                    if category not in category_table:
                        category_table[category] = counter = counter + 1
                
                
            else:
                
                for line in arff_fp:
                    
                    #print (line)
                    #ignore empty lines
                    if len(line)==0:
                        continue
                    if line[0][0]=='@':
                        continue #ignore header lines
                    
                    #remove the curly brace
                    line = line.replace("{","")
                    line = line.replace("}","")
                    
                    
                    #*rest, category = line
                    
                    fields = line.strip().split(ARFF_DELIMITER)
                    
                    if len(fields)<2:
                        continue
                    category = fields[-1].split()[1]
                    
                    rest = fields[:-1]
                    if category not in category_table:
                        numeric_category = category_table[category] = counter = counter + 1
                    else:
                        numeric_category = category_table[category]
                    
                    values = ""
                    for i in range(len(rest)):
                       if i==0:
                           continue
                       index, value = rest[i].split()
                       value = int(value)+addn
                       values+="%s:%s"%(index, value)+SVM_DELIMITER
                   
                    svm_fp.write("%s %s\n" % (numeric_category, values))
        
            return category_table
        
def transform(arff_fp, svm_fp, svm_exist, addn):
            """Transform every training instance of ARFF file to SVM instances
               and return all the field mappings collected."""
            #reader = csv.reader(arff_fp, delimiter=ARFF_DELIMITER)
            ARFF_DELIMITER = ','
            SVM_DELIMITER = ' '
            category_table = {}
            counter = 1
            if svm_exist:
                for line in svm_fp:
                    fields = line.strip().split(SVM_DELIMITER)
                    
                    if len(fields)<2:
                        continue
                    category = fields[0]
                    
                    rest = fields[:-1]
                    if category not in category_table:
                        category_table[category] = counter = counter + 1
            else:
                for line in arff_fp:
                    
                    #print (line)
                    #ignore empty lines
                    if len(line)==0:
                        continue
                    if line[0][0]=='@':
                        continue #ignore header lines
                    
                    
                    #*rest, category = line
                    fields = line.strip().split(ARFF_DELIMITER)
                    category = fields[-1]
                    
                    rest = fields[:-1]
                    if category not in category_table:
                        numeric_category = category_table[category] = counter = counter + 1
                    else:
                        numeric_category = category_table[category]
                   
                    values = SVM_DELIMITER.join("%s:%s"%(i, int(s)+addn)
                        for i, s in enumerate(rest, start=1) if float(s)!=0.0)
                    
                    svm_fp.write("%s %s\n" % (numeric_category, values))
        
            return category_table
    
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Find duplicate accounts of a user')
   
    # Add argument for the dataset path
    parser.add_argument('-i', '--input', help='input file path (.arff or .svm)')
    parser.add_argument('-o', '--output', help='.csv file')
    parser.add_argument('-s', '--models_directory', help='director for saving the classifiers')

    # Parse arguments
    args = parser.parse_args()
    input_arff_file = args.input
    outfile = args.output   
    models_directory = args.models_directory
    
    if  input_arff_file is None or outfile is None:
        parser.print_help()
        exit(1)
        
    if not os.path.exists(models_directory):
        os.mkdir(models_directory)
    
    if not models_directory.endswith('/'):
        models_directory+='/'
    clfname = '100-w10-classifier.joblib.pkl'
    findDoppelgangers(input_arff_file, outfile,  10, True, 4, models_directory, clfname)
          
    