cd $HOME

module purge
module load 2024
module load Anaconda3/2024.06-1

if ! command -v conda --version &> /dev/null
then
    echo "Manually installing Conda as it cannot be found!"
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O conda.sh
    bash conda.sh -b
    echo "y" | conda update --all
    rm conda.sh
fi

cd PersonalizedCIR
if conda info --envs | grep -q ir2; 
then echo "Environment already exists"; 
else
conda create -n ir2 python=3.11 -y
source activate base
conda activate ir2
pip install -r requirements.txt
pip install accelerate
conda install -c pytorch -c nvidia faiss-gpu=1.9.0 -y
conda install -c conda-forge openjdk=21 maven -y
pip install pyserini --upgrade --force-reinstall
pip install -e .;
fi