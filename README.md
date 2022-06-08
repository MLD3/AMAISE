# AMAISE
This contains 
1) the source code for AMAISE: A Machine Learning Approach to Index-Free Sequence Enrichment
2) the demo data to run the source code (in "demo_test_data")
3) the accession codes for the data used to train and test AMAISE (in "accessions.tar.gz")
____________________________________________________________________________________________
AMAISE (A Machine Learning Approach to Index-Free Sequence Enrichment) is a novel, index-free host depletion tool. Given a set of single-end reads, for each sequence, AMAISE outputs a classification label determining whether the sequence is from a host or a microbe (0 for microbe, 1 for host). AMAISE then stores all the microbial sequences in a file for downstream analysis

____________________________________________________________________________________________
## System Requirements

AMAISE requires a computational environment with a Linux-based machine/Ubuntu that has a GPU and enough RAM to support downloading the necessary packages and GPU driver and to support the in-memory operations.

AMAISE also requires pip and Python 3.6 or above. Required packages and versions are listed in "requirements.txt". If your environment is already set up to run code on a GPU, you can install the packages in requirements.txt using:

pip install -r requirements.txt

If your environment is not set up to run code on a GPU, in the folder "helpful_files", I have included a bash script "startup.sh" that contains the code that I used to install GPU drivers onto a Google Cloud Platform Virtual Machine that may help install GPU drivers. "startup.sh" will have to be altered to download the correct drivers for your machine and to set up the appropriate paths to install the packages.

____________________________________________________________________________________________
## Usage Notes for AMAISE

The Data and Code Availability section of the paper contains information needed to alter certain variables that will change the RAM, VRAM, and decision thresholds of AMAISE.

Demo test data is in the folder "demo_test_data." "nanopore_demo_data.fastq" is single-end Nanopore sequences.

The outputs that one should expect from running "host_depletion.py" is in the folder "correct_outputs." You can check the output files beginning with "mlprob" and "mlpath" against the outputs in "correct_outputs" by inputting the output file you generated and the corresponding output file in "correct_outputs" into https://www.diffchecker.com/. You can also check this by using the diff command in linux.

To use AMAISE, use the command

python3 host_depletion.py -i **inputfile** -t **typefile** -o **outfolder**
    
To classify the test data, run
    
python3 host_depletion.py -i demo_test_data/nanopore_demo_data.fastq -t fastq -o single_end_output

Arguments:

i: the reads in a fasta or fastq file
t: the type of file that the reads are in (fasta, fastq)
o: the name of the folder that you want to put the outputs in

Outputs (in **outfolder**): 

mlprobs.txt: a text file with three columns, where the first column is the ID of each read, the second column contains the probabilities that each read is from a host, and the third column is the length of the input sequence

mlpaths.fasta or mlpaths.fastq: a fasta/fastq file (the file type depends on the file type of the input) of all the reads that were classified as microbial
____________________________________________________________________________________________
## Reproducing the Analyses in the Text

The commands you need to be able to run to reproduce the analyses in the text are "time -v", "taskset", "ls -l", and "nvidia-smi." 

To reproduce the analyses in the text on the demo data, you should first run "single_end_AMAISE_evaluation.py" after changing "track_gpu" in "constants.py" to True. This will calculate the evaluation metrics including the amount of VRAM that AMAISE uses and store the calculations. Tracking GPU usage does slow down the code significantly, so to calculate the speed that AMAISE runs in without tracking GPU usage, re-run "single_end_AMAISE_evaluation.py" after setting "track_gpu" in "constants.py" to False. The evaluation metrics are written to "**outfolder**/single_end_resource.txt."

The outputs that one should expect from running "single_end_AMAISE_evaluation.py" is in the folder "correct_outputs." The speed and peak memory usage that you get may differ from the speed and peak memory usage reported in "correct_outputs" since this is dependent on the computational environment that you run the code in. The results in "correct_outputs" are from running the code on a server with 64 AMD EPYC 7702 64-Core Processors (128 hyperthreaded cores), 256 GB of RAM, and 8 NVIDIA RTX 2080 GPUs.

____________________________________________________________________________________________
## Run Time on "Normal" Desktop Computer
    
We mimic a "Normal" Desktop Computer with a Google Cloud Platform Virtual Machine that has
4 vCPUs, 16 GB of RAM, 200 GB of boot disk storage, and 1 Tesla T4 GPU with 320 NVIDIA Turing Tensor Cores and 2560 NVIDIA CUDA cores. 
   
Run Time for Classifying Test Sets: AMAISE takes 44.6 minutes to classify the Nanopore test set.

Run Time for Classifying Demo Data: AMAISE takes 42 seconds to classify the demo Nanopore data.