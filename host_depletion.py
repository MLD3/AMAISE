import numpy as np
import random
import os
import sys, getopt
import torch
import torch.nn as nn
torch.backends.cudnn.enabled=False
from helper import *
from gpu_usage import *
from constants import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# sets up the environment such that AMAISE uses one GPU. Change this to increase the number of 
# GPUs that AMAISE uses
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.set_num_threads(16)

# ensure reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Outputs to help user use AMAISE
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:t:o:')
except getopt.GetoptError:
    print('host_depletion.py -i <inputfile> -t <typefile> -o <outfolder>')
    sys.exit(2)

if len(opts) != 3:
    print('host_depletion.py -i <inputfile> -t <typefile> -o <outfolder>')
    sys.exit()

'''
User Inputs:
inputfile: fasta/fastq file to classify
typefile: 'fastq' if inputfile is a fastq files, and 'fasta' if inputfile is a fasta file
outfolder: folder to write output files to 
'''
for opt, arg in opts:
    if opt == '-h':
        print('host_depletion.py -i <inputfile> -t <typefile> -o <outfolder>')
        sys.exit()
    elif opt in ("-i", "--inputfile"):
        testfile = arg
    elif opt in ("-t", "--typefile"):
        typefile = arg
        if typefile not in ['fastq', 'fasta']:
            print('-t argument: fastq, fasta')
    elif opt in ("-o", "--outfolder"):
        outfolder = arg
            
# Create temp_lenfiles folder
if not os.path.exists('temp_lenfiles'):
    os.makedirs('temp_lenfiles')
    
# Create outfolder folder
if not os.path.exists('%s'%outfolder):
    os.makedirs('%s'%outfolder)
    
# Delete existing information in temp_lenfiles
os.system('rm -f temp_lenfiles/*')

# Delete existing output of AMAISE 
os.system('rm -f %s/mlpaths.%s'%(outfolder, typefile))

# Open file to write probabilities that each input sequence is from a host
outputwritefile = open("%s/mlprobs.txt"%(outfolder), 'w')
outputwritefile.write('id, classification label (0 for microbe, 1 for host), length\n')

# Open file to write microbial sequences to
g = open("%s/mlpaths.%s"%(outfolder, typefile), "w")

# If we are tracking GPU usage, open file to track GPU information
if track_gpu == True:
    gpufile = open('%s/gpuusage.txt'%outfolder, 'w')

# Load AMAISE onto GPUs
model = TCN()
model = nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load(modelpath))
model.eval()

# Get length information to bin input sequences by length
orig_filelens, all_filelens = create_filelens(typefile, testfile)

# Iterate through all bins
for filelens in all_filelens:
    # Delete text files in temp_lenfiles
    os.system('rm -f temp_lenfiles/*')
    
    # Initialize final_size that keeps track of the number of sequences in each length bin
    # that have been classified
    final_size = []
    for i in filelens:
        final_size.append(0)

    # Write sequences that are in the current length bins to "temp_lenfiles"
    sort_seqs(orig_filelens, typefile, testfile, filelens, final_size)
    
    # Iterate through the text files in "temp_lenfiles" and classify the sequences
    with torch.no_grad():
        for len_idx, len_ in enumerate(filelens):
            i = 0
            j = 0
            ids, seqs, lens, y_pred_oh, quals = [], [], [], [], []
            final_len = len_
            
            # Truncate all sequence lengths to "max_len"
            if final_len >= max_len:
                final_len = max_len
                
            # Calculate batch size for testing
            batch_size = int(batch_max_len/final_len)
            temp_batch_size = batch_size
            if os.path.exists('temp_lenfiles/len%d.txt'%len_):
                with open('temp_lenfiles/len%d.txt'%len_) as f:
                    for line in f:
                        # Classify sequences
                        if j == temp_batch_size:
                            X = torch.tensor(X).float().to(device)
                            with torch.cuda.amp.autocast():
                                y_pred_oh = torch.sigmoid(model(X)).detach().cpu().numpy()[:, 1]
                            if track_gpu == True:
                                gpu_usage(gpufile)
                            del X
                            torch.cuda.empty_cache()
                    
                            if typefile == 'fasta':
                                quals = [0]*len(seqs)
                            # Write classifications/ microbial sequences to the appropriate files
                            for id_, pred_, seq_, len_, qual_ in zip(ids, y_pred_oh, seqs, lens, quals):
                                write_output(typefile, outputwritefile, g, id_, seq_, len_, pred_, qual_)
                            ids, seqs, y_pred_oh, lens, quals = [], [], [], [], []
                        # Initialize current batch
                        if i % batch_size == 0:
                            j = 0
                            temp_batch_size = batch_size
                            if i + batch_size > final_size[len_idx]:
                                temp_batch_size = final_size[len_idx] - i
                            X = np.zeros((temp_batch_size, final_len, 4))
                        # Read in information from text files and write it to appropriate variables
                        id_ = line.split(', ')[0]
                        seq_ = line.split(', ')[1]
                        ids.append(id_)            
                        seqs.append(seq_)
                        lens.append(final_len)
                        if typefile == 'fastq':
                            qual_ = line.split(', ')[2].split('\n')[0]
                            quals.append(qual_)
                        
                        # Update array used to store sequence information
                        X[j, :, :] = np.asarray(generate_long_sequences(seq_[:final_len]),  dtype='uint8')
                        i += 1
                        j += 1
                # Classify sequences in final batch
                if i > 0 or j > 0:
                    X = torch.tensor(X).float().to(device)
                    with torch.cuda.amp.autocast():
                        y_pred_oh = torch.sigmoid(model(X)).detach().cpu().numpy()[:, 1]
                    if track_gpu == True:
                        gpu_usage(gpufile)
                    del X
                    torch.cuda.empty_cache()
                    if typefile == 'fasta':
                        quals = [0]*len(seqs)
                    # Write classifications/ microbial sequences to the appropriate files
                    for id_, pred_, seq_, len_, qual_ in zip(ids, y_pred_oh, seqs, lens, quals):
                        write_output(typefile, outputwritefile, g, id_, seq_, len_, pred_, qual_)
                ids, seqs, lens, y_pred_oh, quals = [], [], [], [], []

                # Delete text file we just iterated through in temp_lenfiles
                if os.path.exists('temp_lenfiles/len%d.txt'%len_):
                    os.system('rm -f temp_lenfiles/len%d.txt'%len_)
# Close all opened files                    
outputwritefile.close()
g.close()
if track_gpu == True:
    gpufile.close()

# Delete existing information in temp_lenfiles
os.system('rm -f temp_lenfiles/*')