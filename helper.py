import numpy as np
import torch
import random
import torch.nn as nn
from Bio import SeqIO
from Bio.SeqIO.QualityIO import FastqGeneralIterator
torch.backends.cudnn.enabled=False
from constants import *
import joblib

# ensure reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# the thresholds used to convert AMAISE's output probabilities into classification labels
threshs = {25: 0.31313131313131315, 50: 0.4141414141414142, 100: 0.5454545454545455, 150: 0.6262626262626263, 200: 0.7070707070707072, 250: 0.6363636363636365, 300: 0.6666666666666667, 500: 0.6464646464646465, 1000: 0.4747474747474748, 5000: 0.48484848484848486, 10000: 0.4646464646464647}

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

# the name of the saved model corresponding to the single-end version of AMAISE
modelpath = 'models_and_references/single_end_model'

'''
Inputs:
sequence: A sequence in the form 'ATGC...' (a string of As, Ts, Cs, and Gs)

Outputs:
features: one-hot-encoded version of sequence

generate_long_sequences converts a DNA in string form to a numerical matrix
'''
def generate_long_sequences(sequence):
    """Convert sequence into series of onehot vectors."""
    """If fixed_seq == True longer sequences will be truncated to seq_size
    and shorter sequences will be prepadded with zeros."""
    features = np.zeros((len(sequence), 4))
    seq_list = np.array(list(sequence))
    features[seq_list=="A", 0] = 1
    features[seq_list=="C", 1] = 1
    features[seq_list=="G", 2] = 1
    features[seq_list=="T", 3] = 1
    features[seq_list=="N", 0] = 0.25
    features[seq_list=="N", 1] = 0.25
    features[seq_list=="N", 2] = 0.25
    features[seq_list=="N", 3] = 0.25
    return features

'''
Inputs:
none

Outputs:
TCN: AMAISE's architecture, which consists of 4 convolutional layers, a global average pooling layer, and 1 fully connected layer. Each convolutional layer in AMAISE contains 128 filters of length 15. We applied a rectified-linear unit activation function and an average pooling layer of length 5 after each convolutional layer.

The class TCN contains AMAISE's architecture
'''
class TCN(nn.Module):
    def __init__(self):
        num_input_channels = 4
        num_output_channels = 128
        filter_size = 15
        num_classes = 2
        pool_amt = 5
        
        super().__init__()
        self.c_in1 = nn.Conv1d(num_input_channels, num_output_channels, kernel_size = filter_size, padding = (filter_size - 1)//2, padding_mode='zeros')
        self.c_in2 = nn.Conv1d(num_output_channels, num_output_channels, kernel_size = filter_size, padding = (filter_size - 1)//2, padding_mode='zeros')
        self.c_in3 = nn.Conv1d(num_output_channels, num_output_channels, kernel_size = filter_size, padding = (filter_size - 1)//2, padding_mode='zeros')
        self.c_in4 = nn.Conv1d(num_output_channels, num_output_channels, kernel_size = filter_size, padding = (filter_size - 1)//2, padding_mode='zeros')
        self.fc = nn.Linear(num_output_channels, num_classes)
        self.pool = nn.AvgPool1d(pool_amt)
        self.pad = nn.ConstantPad1d((pool_amt - 1)//2 + 1, 0)
        
        self.filter_size = filter_size
        self.pool_amt = pool_amt
        
        
    def forward(self, x):
        x = x.transpose(2, 1)
        
        old_shape = x.shape[2]
        if x.shape[2] < self.pool_amt:
            x = self.pad(x)
        new_shape = x.shape[2]
        
        output = self.c_in1(x)
        output = torch.relu(output)
        output = self.pool(output)*(new_shape/old_shape)
        
        old_shape = output.shape[2]
        if output.shape[2] < self.pool_amt:
            output = self.pad(output)
        new_shape = output.shape[2]
                
        output = self.c_in2(output)
        output = torch.relu(output)
        output = self.pool(output)*(new_shape/old_shape)
        
        old_shape = output.shape[2]
        if output.shape[2] < self.pool_amt:
            output = self.pad(output)
        new_shape = output.shape[2]
                
        output = self.c_in3(output)
        output = torch.relu(output)
        output = self.pool(output)*(new_shape/old_shape)
        
        old_shape = output.shape[2]
        if output.shape[2] < self.pool_amt:
            output = self.pad(output)
        new_shape = output.shape[2]
                
        output = self.c_in4(output)
        output = torch.relu(output)
        
        last_layer = nn.AvgPool1d(output.size(2))
        output = last_layer(output).reshape(output.size(0), output.size(1))*(new_shape/old_shape)
        output = self.fc(output)
        return output

'''
Inputs:
typefile: 'fastq' if testfile is a fastq files, and 'fasta' if testfile is a fasta files
testfile: fasta/fastq file of sequences for AMAISE to classify 

Outputs:
orig_filelens: a list containing all the bins of sequence lengths to place the reads in testfile. Starting from the minimum length of the reads in testfile until length seq_cutoff, sequences are binned in increments of inc1. For sequences above length seq_cutoff, the bin size is inc2. 

all_filelens: an array of lists containing the lengths that can be written to memory until "lim" is hit. 

create_filelens_paired returns lists containing the information needed to bin the input sequences by length for efficient classification.
'''
def create_filelens(typefile, testfile):
    num_len = {}
    orig_filelens = []
    if typefile == 'fastq':
        for title, seq, qual in FastqGeneralIterator(open(testfile)):
            if len(seq) < seq_cutoff:
                inc = inc1
            else:
                inc = inc2
            final_i = (len(seq)//inc) * inc
            if final_i in num_len.keys():
                num_len[final_i] += 1
            else:
                num_len[final_i] = 1
                orig_filelens.append(final_i)
    else:
        for record in SeqIO.parse(testfile, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) < seq_cutoff:
                inc = inc1
            else:
                inc = inc2
            final_i = (len(seq)//inc) * inc
            if final_i in num_len.keys():
                num_len[final_i] += 1
            else:
                num_len[final_i] = 1
                orig_filelens.append(final_i)
            num_len[final_i] += 1
    
    orig_filelens.sort()
    all_filelens = []
    curr_filelens = []
    curr_count = 0
    for i in orig_filelens:
        if curr_count + num_len[i]*i > lim:
            all_filelens.append(curr_filelens)
            curr_filelens = []
            curr_count = 0
        if num_len[i] > 0:
            curr_count += num_len[i]*i
            curr_filelens.append(i)
    all_filelens.append(curr_filelens)
    return orig_filelens, all_filelens

'''
Inputs:
orig_filelens: a list containing all the bins of sequence lengths to place the reads in testfile. Starting from the minimum length of the reads in testfile until length seq_cutoff, sequences are binned in increments of inc1. For sequences above length seq_cutoff, the bin size is inc2. 
typefile: 'fastq' if testfile is a fastq file, and 'fasta' if testfile is a fasta file
testfile: fasta/fastq file of sequences for AMAISE to classify 
filelens: a list from all_filelens, which is a list of lists of bin sizes
final_size: a dictionary that keeps track of the number of sequences in each bin that have been written to a text file

Outputs:
files in the folder "temp_lenfiles" that contain the sequences to be classified

sort_seqs_paired bins sequences by length and writes sequences of the same length into a textfile in the folder temp_lenfiles with the name "len<sequence length>.txt"

'''
def sort_seqs(orig_filelens, typefile, testfile, filelens, final_size):
    filelens = np.array(filelens)
    if typefile == 'fastq':
        for title, seq, qual in FastqGeneralIterator(open(testfile)):
            if filelens[-1] < seq_cutoff:
                inc = inc1
            else:
                inc = inc2
            if (len(seq) >= filelens[0] and len(seq) < filelens[-1] + inc) or (filelens[-1] == orig_filelens[-1] and len(seq) > filelens[-1]):
                if len(seq) < filelens[-1]:
                    final_i = np.max(filelens[filelens <= len(seq)])
                    final_idx = np.argmax(filelens[filelens <= len(seq)])
                else:
                    final_i = filelens[-1]
                    final_idx = len(filelens) - 1
                f = open('temp_lenfiles/len%d.txt'%final_i, 'a')
                f.write('%s, %s, %s\n'%(title.split(None,1)[0], str(seq).upper(), str(qual)))
                f.close()
                final_size[final_idx] += 1
    else:
        for record in SeqIO.parse(testfile, "fasta"):
            title = str(record.id)
            seq = str(record.seq).upper()
            if len(seq) < seq_cutoff:
                inc = inc1
            else:
                inc = inc2
            if (len(seq) >= filelens[0] and len(seq) < filelens[-1] + inc) or (filelens[-1] == orig_filelens[-1] and len(seq) > filelens[-1]):
                if len(seq) < filelens[-1]:
                    final_i = np.max(filelens[filelens <= len(seq)])
                    final_idx = np.argmax(filelens[filelens <= len(seq)])
                else:
                    final_i = filelens[-1]
                    final_idx = len(filelens) - 1
                if final_i == 0:
                    print(len(seq))
                f = open('temp_lenfiles/len%d.txt'%final_i, 'a')
                f.write('%s, %s\n'%(title, str(seq).upper()))
                f.close()
                final_size[final_idx] += 1
    
'''
Inputs: 
typefile: 'fastq' if testfile is a fastq file, and 'fasta' if testfile is a fasta file
outputwritefile: the file to write the classification label corresponding to each read (0 for microbe, 1 for host)
g: the file to write microbial reads to
id_: the sequence id of the current read being written to outputwritefile and g
seq_: the base pairs of the sequence corresponding to the id "id_"
pred_: the probability that the read is from a host
qual_: if the input read is from a fast2 file, the quality scores of the read. If the read is from a fasta file, this is 0.

Outputs:
g updated with the input read if it is from a microbe and outputwritefile updated with the classification label of the read

write_output_paired writes the classification label of the single-end read in variable "seq_" with sequence id "id_", and writes its sequence information into the fasta/fastq file g

'''
def write_output(typefile, outputwritefile, g, id_, seq_, len_, pred_, qual_):
    final_len = closest(list(threshs.keys()), len_)
    label = 1
    if pred_ < threshs[final_len]:
        label = 0
    outputwritefile.write('%s, %d, %d\n'%(id_, label, len_))
    if pred_ < threshs[final_len]:
        if typefile == 'fastq':
            g.write('@%s\n'%id_)
        else:
            g.write('>%s\n'%id_)
        g.write('%s\n'%seq_)
        if typefile == 'fastq':
            g.write('+\n')
            g.write('%s\n'%qual_)
