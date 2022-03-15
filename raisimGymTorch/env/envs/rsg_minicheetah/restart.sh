#!/bin/bash
#wmctrl -c "Firefox" -x "Navigator.Firefox"
#set -x

#conda run -n jakob python runner.py --runNumber=0
conda run -n jakob python runner.py
cd /home/jakob/raisim_workspace/raisimLib/raisimGymForRaisin
conda run -n jakob python setup.py develop
cd /home/jakob/raisim_workspace/raisimLib/raisimGymForRaisin/raisimGymTorch/env/envs/rsg_minicheetah/
conda run -n jakob python runner.py
#conda run -n jakob python runner.py --runNumber=1
#conda run -n jakob python runner.py --runNumber=2
#conda run -n jakob python runner.py --runNumber=3
#conda run -n jakob python runner.py --runNumber=4
#conda run -n jakob python runner.py --runNumber=5

#for i in {1..8}
#do
#   conda run -n jakob python runner.py --runNumber=0
#done