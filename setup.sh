#!/usr/bin/env bash

conda create --name 'var_comp' python==3.10
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate var_comp
pip install -r requirements.txt