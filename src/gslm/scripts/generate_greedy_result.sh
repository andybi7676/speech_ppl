set -e 

root_dir=~/speech_ppl
cd $root_dir
source $root_dir/textlesslib/.venv/bin/activate

# TODO: move some hard-coded vars to args
python src/gslm/tools/gslm_speech_ppl_wrapper.py