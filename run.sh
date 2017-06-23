#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export GLOG_minloglevel=2

core_num=`nproc`

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=${core_num}
export MKL_NUM_THREADS=${core_num}
# export OMP_DYNAMIC="False"

# export OMP_WAIT_POLICY=passive
unset MKL_THREADING_LAYER
# export MKL_THREADING_LAYER=gnu
export KMP_AFFINITY=compact,1,0,granularity=fine

./experiments/scripts/rfcn_end2end.sh 0 ResNet-101 pascal_voc
