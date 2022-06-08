import subprocess
import datetime

'''
Input: 
gpufile: an opened file that has write permissions on

Output:
text written to gpufile

gpu_usage records the amount of VRAM that AMAISE uses
'''
def gpu_usage(gpufile):
    output = subprocess.check_output(['nvidia-smi'], shell = True, stderr=subprocess.STDOUT)
    elems = output.decode("utf-8").split()
    mib = []
    for i in elems:
        if 'MiB' in i:
            mib.append(i)
    gpu_usage = mib[0]
    gpufile.write('%s\n'%gpu_usage)
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    gpufile.write("Current Time =%s\n"%current_time)
   