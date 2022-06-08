import os
import subprocess
from accuracy import *
import joblib

# define the file with the true labels and the file with the corresponding DNA sequences
truelabels = 'demo_test_data/nanopore_demo_data.txt'
truefile = truelabels
inputfile = 'demo_test_data/nanopore_demo_data.fastq'

# define the model used to classify the input sequences
modelpath = 'models_and_references/single_end_model'

# define the thresholds used to convert AMAISE's output probabilities into classification labels
threshs = {25: 0.31313131313131315, 50: 0.4141414141414142, 100: 0.5454545454545455, 150: 0.6262626262626263, 200: 0.7070707070707072, 250: 0.6363636363636365, 300: 0.6666666666666667, 500: 0.6464646464646465, 1000: 0.4747474747474748, 5000: 0.48484848484848486, 10000: 0.4646464646464647}

'''
Inputs: 
inputfile: 

Outputs:
outputfolder:
resourcefile:

ml calculates the ...
'''
def ml(inputfile):
    typefile = 'fastq'
    outputfolder = 'single_end_output'
    resourcefile = '%s/single_end_resource.txt'%outputfolder
    # get the elapsed wall clock time and peak memory usage of running AMAISE
    cmd = 'time -v taskset -c 0 python3 host_depletion.py -i %s -t %s -o %s'%(inputfile, typefile, outputfolder)
    print(cmd)
    output = subprocess.check_output([cmd], shell = True, stderr=subprocess.STDOUT)
    elems = output.decode("utf-8")
    for i in range(len(elems.split('\n'))):
        if 'Elapsed (wall clock) time' in elems.split('\n')[i]:
            print(elems.split('\n')[i].split('\t')[1])
            mlspeed = elems.split('\n')[i].split('\t')[1].split(':')[-3:]
            try: 
                mlspeed = int(mlspeed[0])*60*60 + float(mlspeed[1])*60 + float(mlspeed[2])
            except:
                mlspeed = float(mlspeed[1])*60 + float(mlspeed[2])
        if 'Maximum resident set size' in elems.split('\n')[i]:
            size_line = elems.split('\n')[i].split('\t')[1]
            size = int(size_line.split(': ')[1])
            print('Maximum resident set size (GB): %0.4f'%(size/(10**6)))
            mlressetsize = size/(10**6)

    # get the total storage needed to run AMAISE
    output = subprocess.check_output(['ls -l %s'%modelpath], shell = True, stderr=subprocess.STDOUT)
    elems = output.decode("utf-8").split('meerak ')
    total_storage = 0
    for elem in elems:
        inside_elem = elem.split()
        if len(inside_elem) > 2:
            total_storage += int(inside_elem[0])
    print('Total storage (GB): %0.4f'%(total_storage/(10**9)))
    mltotalstorage = total_storage/(10**9)

    # get the accuracy, sensitivity, and specificity from running AMAISE
    accuracy, sens, spec = ml_rhd(truelabels, '%s/mlprobs.txt'%outputfolder, threshs)
    print(accuracy, sens, spec)
    with open(resourcefile, 'w') as f:
        f.write('Speed: %0.10f\n'%mlspeed)
        f.write('Peak Memory Usage: %0.10f\n'%mlressetsize)
        f.write('Total Storage: %0.10f\n'%mltotalstorage)
        f.write('Accuracy: %0.10f\n'%accuracy)
        f.write('Sensitivity: %0.10f\n'%sens)
        f.write('Specificity: %0.10f\n'%spec)

ml(inputfile)
