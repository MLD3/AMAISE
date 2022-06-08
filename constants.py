# When AMAISE is evaluating single-end reads, starting from the minimum length sequence until length seq_cutoff, sequences are binned in increments of inc1 and sequences within the same bin are saved to a text file and truncated to the smallest sequence in that bin. 
seq_cutoff = 5000
inc1 = 50

# For sequences above length seq_cutoff, the bin size is inc2. 
inc2 = 1000

# The total sequence length that AMAISE can write to memory is lim.
lim = 400000000*3

# All sequences that are greater in length than max_len are truncated to length max_len. 
max_len = 6000

# The batch size that we use for testing is batch_max_len/L where L is the length of the sequences in the batch 
batch_max_len = 900000

# When track_gpu is True, the MiB that AMAISE uses of VRAM is stored in the text file gpuusage.txt  
track_gpu = True