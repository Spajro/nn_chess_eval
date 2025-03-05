conda create -n nnue
conda activate nnue
conda init
conda install chess
conda install zstandard
conda install stockfish
conda install pytorch==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda deactivate