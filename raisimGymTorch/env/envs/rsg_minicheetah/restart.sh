#!/bin/bash
#wmctrl -c "Firefox" -x "Navigator.Firefox"
#set -x

conda run -n jakob python runner.py --runNumber=0
conda run -n jakob python runner.py --runNumber=1
conda run -n jakob python runner.py --runNumber=2

#for i in {1..8}
#do
#   conda run -n jakob python runner.py --runNumber=0
#done