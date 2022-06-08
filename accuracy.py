from Bio import SeqIO
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import joblib
import random

# Set seed to ensure deterministic output
seed = 0
random.seed(seed)
np.random.seed(seed)


'''
Inputs:
lst: list, list of lengths whose optimal threshold has been calculated
K: single number, length of sequence whose optimal threshold is to be determined

Outputs:
single number, the element in lst that is closest to K

closest determines the element in the input list "lst" that is closest to the integer
'''
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

'''
seqid2taxid is a dictionary whose values are sequence ids and the keys are the taxonomy id
corresponding to the sequence id
'''
seqid2taxid = {}
with open('models_and_references/seqid2taxid_abfpvHost.txt', 'r') as f:
    for line in f:
        seqid2taxid[line.split('\t')[1][:-1]] = line.split('\t')[0]
 
'''
hosts contains all the species to be considered host
'''
hosts = ['Homo sapiens', 'Mus musculus', 'Sus scrofa']

    
'''
dict_ref is a dictionary whose keys are sequence ids and whose values are the species corresponding to that sequence id
'''
dict_ref = joblib.load('models_and_references/dict_ref')


'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of Kraken2-H containing Kraken2-H's predicted label for all sequence ids
in the text file

Outputs:
accuracy: Kraken2-H's accuracy at separating host from microbial sequences
sensitivity: Kraken2-H's sensitivity at separating host from microbial sequences
specificity: Kraken2-H's specificity at separating host from microbial sequences

kraken_rhd takes in the true labels of a set of DNA sequences and their predicted labels from Kraken2-H (Kraken2 with only host DNA in its index) and outputs the accuracy, sensitivity, and specificity of Kraken2-H at separating host from microbial sequences
'''
def kraken_rhd(truefile, predfile):
    true_dict = {}
    with open(truefile, 'r') as f:
        for line in f:
            if line.split(', ')[1][:-1] in hosts:
                true_dict[line.split(', ')[0]] = 1
            else:
                true_dict[line.split(', ')[0]] = 0
                
    pred_dict = {}
    with open(predfile, 'r') as f:
        for line in f:
            taxid = line.split('\t')[2]
            if taxid in seqid2taxid.keys():
                seqid = seqid2taxid[taxid]
                label = dict_ref[seqid]
                if label in hosts:
                    pred_dict[line.split('\t')[1]] = 1
                else:
                    pred_dict[line.split('\t')[1]] = 0
            else:
                pred_dict[line.split('\t')[1]] = 0

    true = []
    preds = []
    for i in list(true_dict.keys()):
        true.append(true_dict[i])
        preds.append(pred_dict[i])  
    accuracy = accuracy_score(true, preds)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    sens =  tp/(tp + fn)
    spec = tn/(tn + fp)
    return accuracy, sens, spec

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of Centrifuge-H containing Centrifuge-H's predicted label for all sequence ids
in the text file

Outputs:
accuracy: Centrifuge-H's accuracy at separating host from microbial sequences
sensitivity: Centrifuge-H's sensitivity at separating host from microbial sequences
specificity: Centrifuge-H's specificity at separating host from microbial sequences

centrifuge_rhd takes in the true labels of a set of DNA sequences and their predicted labels from Centrifuge-H (Centrifuge with only host DNA in its index) and outputs the accuracy, sensitivity, and specificity of Centrifuge-H at separating host from microbial sequences
'''
def centrifuge_rhd(truefile, predfile):
    true_dict = {}
    with open(truefile, 'r') as f:
        for line in f:
            if line.split(', ')[1][:-1] in hosts:
                true_dict[line.split(', ')[0]] = 1
            else:
                true_dict[line.split(', ')[0]] = 0

    pred_dict = {}
    count = 0
    with open(predfile, 'r') as f:
        for line in f:
            if count > 0:
                taxid = line.split('\t')[2]
                if taxid in seqid2taxid.keys():
                    seqid = seqid2taxid[taxid]
                    label = dict_ref[seqid]
                    if label in hosts:
                        pred_dict[line.split('\t')[0]] = 1
                    else:
                        pred_dict[line.split('\t')[0]] = 0
                else:
                    pred_dict[line.split('\t')[0]] = 0
            count += 1

    true = []
    preds = []
    print(len(true_dict.keys()), len(pred_dict.keys()))
    for i in list(true_dict.keys()):
        if i in true_dict.keys():
            true.append(true_dict[i])
        elif i + '/1' in true_dict.keys():
            true.append(true_dict[i+'/1'])
        preds.append(pred_dict[i])  
    accuracy = accuracy_score(true, preds)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    sens =  tp/(tp + fn)
    spec = tn/(tn + fp)
    return accuracy, sens, spec


'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of Bowtie2-H containing Bowtie2-H's predicted label for all sequence ids
in the text file

Outputs:
accuracy:  Bowtie2-H's accuracy at separating host from microbial sequences
sensitivity:  Bowtie2-H's sensitivity at separating host from microbial sequences
specificity:  Bowtie2-H's specificity at separating host from microbial sequences

bowtie_rhd takes in the true labels of a set of DNA sequences and their predicted labels from Bowtie2-H (Bowtie2 with only host DNA in its index) and outputs the accuracy, sensitivity, and specificity of Bowtie2-H at separating host from microbial sequences
'''
def bowtie_rhd(truefile, predfile):
    true_dict = {}
    with open(truefile, 'r') as f:
        for line in f:
            if line.split(', ')[1][:-1] in hosts:
                true_dict[line.split(', ')[0]] = 1
            else:
                true_dict[line.split(', ')[0]] = 0

    pred_dict = {}
    with open(predfile, 'r') as f:
        for line in f:
            if line[0] != '@':
                input_seq = line.split()[0]
                rec_id = line.split()[2]
                pred_dict[input_seq] = 1
                
    true = []
    preds = []
    for i in list(true_dict.keys()):
        true.append(true_dict[i])
        if i not in pred_dict.keys():
            preds.append(0)  
        else:
            preds.append(pred_dict[i])  
        
    accuracy = accuracy_score(true, preds)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    sens =  tp/(tp + fn)
    spec = tn/(tn + fp)

    return accuracy, sens, spec

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of Minimap2-H containing Minimap2-H's predicted label for all sequence ids
in the text file

Outputs:
accuracy:  Minimap2-H's accuracy at separating host from microbial sequences
sensitivity:  Minimap2-H's sensitivity at separating host from microbial sequences
specificity:  Minimap2-H's specificity at separating host from microbial sequences

minimap_rhd takes in the true labels of a set of DNA sequences and their predicted labels from Minimap2-H (Minimap2 with only host DNA in its index) and outputs the accuracy, sensitivity, and specificity of Minimap2-H at separating host from microbial sequences
'''
def minimap_rhd(truefile, predfile):
    true_dict = {}
    with open(truefile, 'r') as f:
        for line in f:
            if line.split(', ')[1][:-1] in hosts:
                true_dict[line.split(', ')[0]] = 1
            else:
                true_dict[line.split(', ')[0]] = 0

    pred_dict = {}
    with open(predfile, 'r') as f:
        for line in f:
            if line[0] != '@':
                input_seq = line.split()[0]
                rec_id = line.split()[2]
                if rec_id != '*' and rec_id in dict_ref.keys():
                    pred_dict[input_seq] = 1
                else:
                    pred_dict[input_seq] = 0
    true = []
    preds = []
    for i in list(true_dict.keys()):
        true.append(true_dict[i])
        preds.append(pred_dict[i])  
    accuracy = accuracy_score(true, preds)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    sens =  tp/(tp + fn)
    spec = tn/(tn + fp)
    
    return accuracy, sens, spec

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of AMAISE containing the probabilities of each sequence being from a host

Outputs:
accuracy: AMAISE's accuracy at separating host from microbial sequences
sensitivity: AMAISE's sensitivity at separating host from microbial sequences
specificity: AMAISE's specificity at separating host from microbial sequences

ml_rhd takes in the true labels of a set of DNA sequences and the probabilities that they are from a host from AMAISE and outputs the accuracy, sensitivity, and specificity of AMAISE at separating host from microbial sequences
'''
def ml_rhd(truefile, predfile, threshs):
    true_dict = {}
    with open(truefile, 'r') as f:
        for line in f:
            if line.split(', ')[1][:-1] in hosts:
                true_dict[line.split(', ')[0]] = 1
            else:
                true_dict[line.split(', ')[0]] = 0

    pred_dict = {}
    count = 0
    with open(predfile, 'r') as f:
        for line in f:
            if count > 0:
                seqid = line.split(', ')[0]
                pred = float(line.split(', ')[1])
                final_len = closest(list(threshs.keys()), int(line.split(', ')[2]))
                if pred > threshs[final_len]:
                    pred_dict[seqid] = 1
                else:
                    pred_dict[seqid] = 0
            count += 1

    true = []
    preds = []
    for i in list(true_dict.keys()):
        true.append(true_dict[i])
        preds.append(pred_dict[i])  
    accuracy = accuracy_score(true, preds)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    sens =  tp/(tp + fn)
    spec = tn/(tn + fp)

    return accuracy, sens, spec

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of Kraken2-HM containing Kraken2-HM's predicted label for all sequence ids
ML_outputfile: the output file of AMAISE containing the probabilities of each sequence being from a host
threshs: the thresholds used to convert AMAISE's output probabilities into classification labels

Outputs:
overall_acc: the accuracy of Kraken2-HM at classifying the test sequences
filter_acc: the accuracy of Kraken2-HM at classifying the sequences AMAISE classified as microbial
unfilter_acc: the accuracy of Kraken2-HM at classifying the sequences AMAISE classified as microbial
host_acc: the accuracy of Kraken2-HM at classifying the sequences whose true label is host
microbe_acc: the accuracy of Kraken2-HM at classifying the sequences whose true label is microbial

kraken_acc takes in the true labels of a set of DNA sequences and their predicted labels from Kraken2-HM and outputs Kraken2-HM's overall accuracy, Kraken2-HM's accuracy on the sequences AMAISE classified as microbial, Kraken2-HM's accuracy on the sequences AMAISE classified as host, Kraken2-HM's accuracy on the sequences whose true label is microbial, and Kraken2-HM's accuracy on the sequences whose true label is host

host_acc and microbe_acc are reported in the paper
'''    
def kraken_acc(truefile, predfile, ML_outputfile, threshs):
    filter_ids = []
    unfilter_ids = []
    count = 0
    with open(ML_outputfile, 'r') as f:
        for line in f:
            if count > 0:
                seqid = line.split(', ')[0]
                pred = float(line.split(', ')[1])
                final_len = closest(list(threshs.keys()), int(line.split(', ')[2]))
                if pred <= threshs[final_len]:
                    filter_ids.append(seqid)
                else:
                    unfilter_ids.append(seqid)
            count += 1
    
    true_dict = {}
    unique_labels = []
    with open(truefile, 'r') as f:
        for line in f:
            true_dict[line.split(', ')[0]] = line.split(', ')[1][:-1].split(',')[0]
            unique_labels.append(line.split(', ')[1][:-1].split(',')[0])
    
    unique_labels = list(set(unique_labels))
    unique_labels.append('None')    
    unique_labels_dict = {}
    for i in range(len(unique_labels)):
        unique_labels_dict[unique_labels[i]] = i

    pred_dict = {}
    with open(predfile, 'r') as f:
        for line in f:
            taxid = line.split('\t')[2]
            if taxid in seqid2taxid.keys():
                seqid = seqid2taxid[taxid]
                label = dict_ref[seqid]
                pred_dict[line.split('\t')[1]] = label
                if label not in unique_labels:
                    pred_dict[line.split('\t')[1]] = 'None'
            else:
                pred_dict[line.split('\t')[1]] = 'None'
    
    overall_count = len(filter_ids) + len(unfilter_ids)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    overall_acc = accuracy_score(true, preds)
    
    
    true = []
    preds = []
    for i in filter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    filter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in unfilter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    unfilter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    host_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] not in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    microbe_acc = accuracy_score(true, preds)
    
    return overall_acc, filter_acc, unfilter_acc, host_acc, microbe_acc

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile: the output file of Centrifuge-HM containing Centrifuge-HM's predicted label for all sequence ids
ML_outputfile: the output file of AMAISE containing the probabilities of each sequence being from a host
threshs: the thresholds used to convert AMAISE's output probabilities into classification labels

Outputs:
overall_acc: the accuracy of Centrifuge-HM at classifying the test sequences
filter_acc: the accuracy of Centrifuge-HM at classifying the sequences AMAISE classified as microbial
unfilter_acc: the accuracy of Centrifuge-HM at classifying the sequences AMAISE classified as microbial
host_acc: the accuracy of Centrifuge-HM at classifying the sequences whose true label is host
microbe_acc: the accuracy of Centrifuge-HM at classifying the sequences whose true label is microbial

centrifuge_acc takes in the true labels of a set of DNA sequences and their predicted labels from Centrifuge-HM and outputs Centrifuge-HM's overall accuracy, Centrifuge-HM's accuracy on the sequences AMAISE classified as microbial, Centrifuge-HM's accuracy on the sequences AMAISE classified as host, Centrifuge-HM's accuracy on the sequences whose true label is microbial, and Centrifuge-HM's accuracy on the sequences whose true label is host

host_acc and microbe_acc are reported in the paper
'''   
def centrifuge_acc(truefile, predfile, ML_outputfile, threshs):
    count = 0
    filter_ids = []
    unfilter_ids = []
    with open(ML_outputfile, 'r') as f:
        for line in f:
            if count > 0:
                seqid = line.split(', ')[0]
                pred = float(line.split(', ')[1])
                final_len = closest(list(threshs.keys()), int(line.split(', ')[2]))
                if pred <= threshs[final_len]:
                    filter_ids.append(seqid)
                else:
                    unfilter_ids.append(seqid)
            count += 1
            
    true_dict = {}
    unique_labels = []
    with open(truefile, 'r') as f:
        for line in f:
            true_dict[line.split(', ')[0]] = line.split(', ')[1][:-1].split(',')[0]
            unique_labels.append(line.split(', ')[1][:-1].split(',')[0])

    unique_labels = list(set(unique_labels))
    unique_labels.append('None')  
    unique_labels_dict = {}
    for i in range(len(unique_labels)):
        unique_labels_dict[unique_labels[i]] = i

    pred_dict = {}
    count = 0
    with open(predfile, 'r') as f:
        for line in f:
            if count > 0:
                taxid = line.split('\t')[2]
                if taxid in seqid2taxid.keys():
                    seqid = seqid2taxid[taxid]
                    label = dict_ref[seqid]
                    pred_dict[line.split('\t')[0]] = label
                    if label not in unique_labels:
                        pred_dict[line.split('\t')[0]] = 'None'
                else:
                    pred_dict[line.split('\t')[0]] = 'None'
            count += 1
    
    overall_count = len(filter_ids) + len(unfilter_ids)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    overall_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in filter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    filter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in unfilter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    unfilter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    host_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] not in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    microbe_acc = accuracy_score(true, preds)
    
    return overall_acc, filter_acc, unfilter_acc, host_acc, microbe_acc

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile1: the output file of AMAISE containing the probabilities of each sequence being from a host
predfile2: the output file of Kraken2-M containing the Kraken2-M's predicted label for the sequences AMAISE labeled as microbial
threshs: the thresholds used to convert AMAISE's output probabilities into classification labels

Outputs:
overall_acc: the accuracy of AMAISE + Kraken2-M at classifying the test sequences
filter_acc: the accuracy of AMAISE + Kraken2-M at classifying the sequences AMAISE classified as microbial
unfilter_acc: the accuracy of AMAISE + Kraken2-M at classifying the sequences AMAISE classified as microbial
host_acc: the accuracy of AMAISE + Kraken2-M at classifying the sequences whose true label is host
microbe_acc: the accuracy of AMAISE + Kraken2-M at classifying the sequences whose true label is microbial

krakenML_acc_mults takes in the true labels of a set of DNA sequences, AMAISE's output, and Kraken2-M's output given the sequences that AMAISE classified as microbial and outputs AMAISE + Kraken2-M's overall accuracy, AMAISE + Kraken2-M's accuracy on the sequences AMAISE classified as microbial, AMAISE + Kraken2-M's accuracy on the sequences AMAISE classified as host, AMAISE + Kraken2-M's accuracy on the sequences whose true label is microbial, and AMAISE + Kraken2-M's accuracy on the sequences whose true label is host

host_acc and microbe_acc are reported in the paper
'''   
def krakenML_acc_mults(truefile, predfile1, predfile2, threshs):
    true_dict = {}
    unique_labels = []
    with open(truefile, 'r') as f:
        for line in f:
            true_dict[line.split(', ')[0]] = line.split(', ')[1][:-1].split(',')[0]
            unique_labels.append(line.split(', ')[1][:-1].split(',')[0])

    unique_labels = list(set(unique_labels))
    unique_labels.append('None')  
    unique_labels_dict = {}
    for i in range(len(unique_labels)):
        unique_labels_dict[unique_labels[i]] = i

    count = 0
    pred_dict = {}
    filter_ids = []
    unfilter_ids = []
    with open(predfile1, 'r') as f:
        for line in f:
            if count > 0:
                seqid = line.split(', ')[0]
                pred = float(line.split(', ')[1])
                final_len = closest(list(threshs.keys()), int(line.split(', ')[2]))
                if pred > threshs[final_len]:
                    pred_dict[seqid] = 'Homo sapiens'
                    unfilter_ids.append(seqid)
                else:
                    filter_ids.append(seqid)
            count += 1
    
    with open(predfile2, 'r') as f:
        for line in f:
            taxid = line.split('\t')[2]
            if taxid in seqid2taxid.keys():
                seqid = seqid2taxid[taxid]
                label = dict_ref[seqid]
                pred_dict[line.split('\t')[1]] = label
                if label not in unique_labels:
                    pred_dict[line.split('\t')[1]] = 'None'
            else:
                pred_dict[line.split('\t')[1]] = 'None'
    
    overall_count = len(filter_ids) + len(unfilter_ids)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    overall_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in filter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    filter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in unfilter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    unfilter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    host_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] not in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    microbe_acc = accuracy_score(true, preds)
    
    return overall_acc, filter_acc, unfilter_acc, host_acc, microbe_acc

'''
Inputs:
truefile: a file containing the sequence ids of the test file and their true labels
predfile1: the output file of AMAISE containing the probabilities of each sequence being from a host
predfile2: the output file of Centrifuge-M containing the Centrifuge-M's predicted label for the sequences AMAISE labeled as microbial
threshs: the thresholds used to convert AMAISE's output probabilities into classification labels

Outputs:
overall_acc: the accuracy of AMAISE + Centrifuge-M at classifying the test sequences
filter_acc: the accuracy of AMAISE + Centrifuge-M at classifying the sequences AMAISE classified as microbial
unfilter_acc: the accuracy of AMAISE + Centrifuge-M at classifying the sequences AMAISE classified as microbial
host_acc: the accuracy of AMAISE + Centrifuge-M at classifying the sequences whose true label is host
microbe_acc: the accuracy of AMAISE + Centrifuge-M at classifying the sequences whose true label is microbial

centrifugeML_acc_mults takes in the true labels of a set of DNA sequences, AMAISE's output, and Centrifuge-M's output given the sequences that AMAISE classified as microbial and outputs AMAISE + Centrifuge-M's overall accuracy, AMAISE + Centrifuge-M's accuracy on the sequences AMAISE classified as microbial, AMAISE + Centrifuge-M's accuracy on the sequences AMAISE classified as host, AMAISE + Centrifuge-M's accuracy on the sequences whose true label is microbial, and AMAISE + Centrifuge-M's accuracy on the sequences whose true label is host

host_acc and microbe_acc are reported in the paper
'''  
def centrifugeML_acc_mults(truefile, predfile1, predfile2, threshs):
    true_dict = {}
    unique_labels = []
    with open(truefile, 'r') as f:
        for line in f:
            true_dict[line.split(', ')[0]] = line.split(', ')[1][:-1].split(',')[0]
            unique_labels.append(line.split(', ')[1][:-1].split(',')[0])
    
    unique_labels = list(set(unique_labels))
    unique_labels.append('None')  

    unique_labels_dict = {}
    for i in range(len(unique_labels)):
        unique_labels_dict[unique_labels[i]] = i
        
    count = 0
    pred_dict = {}
    filter_ids = []
    unfilter_ids = []
    with open(predfile1, 'r') as f:
        for line in f:
            if count > 0:
                seqid = line.split(', ')[0]
                pred = float(line.split(', ')[1])
                final_len = closest(list(threshs.keys()), int(line.split(', ')[2]))
                if pred > threshs[final_len]:
                    pred_dict[seqid] = 'Homo sapiens'
                    unfilter_ids.append(seqid)
                else:
                    filter_ids.append(seqid)
            count += 1

    count = 0
    with open(predfile2, 'r') as f:
        for line in f:
            if count > 0:
                taxid = line.split('\t')[2]
                if taxid in seqid2taxid.keys():
                    seqid = seqid2taxid[taxid]
                    label = dict_ref[seqid]
                    pred_dict[line.split('\t')[0]] = label
                    if label not in unique_labels:
                        pred_dict[line.split('\t')[0]] = 'None'
                else:
                    pred_dict[line.split('\t')[0]] = 'None'
            count += 1
    
    overall_count = len(filter_ids) + len(unfilter_ids)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    overall_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in filter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    
    filter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in unfilter_ids:
        if true_dict[i] in hosts and pred_dict[i] in hosts:
            pred_dict[i] = true_dict[i]
        true.append(unique_labels_dict[true_dict[i]])
        preds.append(unique_labels_dict[pred_dict[i]])
    unfilter_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    host_acc = accuracy_score(true, preds)
    
    true = []
    preds = []
    for i in list(true_dict.keys()):
        if true_dict[i] not in hosts:
            true.append(unique_labels_dict[true_dict[i]])
            preds.append(unique_labels_dict[pred_dict[i]])
    microbe_acc = accuracy_score(true, preds)
    
    return overall_acc, filter_acc, unfilter_acc, host_acc, microbe_acc

