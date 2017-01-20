export GLOG_minloglevel=2


# export OMP_NUM_THREADS
# export MKL_NUM_THREADS

# export OMP_WAIT_POLICY=passive
# export MKL_THREADING_LAYER=gnu


./experiments/scripts/rfcn_end2end.sh 0 ResNet-101 pascal_voc
