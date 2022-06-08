sudo apt install pciutils
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install linux-headers-$(uname -r)

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
sudo apt-get install software-properties-common

# alter this if you do not have a Debian 10 operating system that runs 64-bit software
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"

sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}


sudo apt-get install python3-distutils
sudo apt-get install python3-apt
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
# alter this path to change meerak to your username
export PATH="/home/meerak/.local/bin/:$PATH"
pip install numpy
pip install torch
pip install biopython
pip install joblib
pip install matplotlib
pip install scikit-learn

nvcc -V
pip install gpustat

sudo apt-get install time

