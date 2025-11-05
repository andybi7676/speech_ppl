set -e 

root_dir=/mnt/storages/nfs/projects/speech_ppl
cd $root_dir
source $root_dir/venv/gslm/.venv/bin/activate

device=cuda
gslm_pretrained_model_dir=$root_dir/work/pretrained_models/gslm/hubert100_lm
data_sample_dir=$root_dir/work/data/samples
gslm_output_dir=$root_dir/work/outputs/gslm
mkdir -p $gslm_output_dir
export CUDA_VISIBLE_DEVICES=0

# test loss calculation and generation on a single sample
# python src/gslm/tools/gslm_speech_ppl_wrapper.py \
#     --language_model_dir $gslm_pretrained_model_dir \
#     --testing_audio_fpath $data_sample_dir/61-70968-0000_orig.flac \
#     --output_dir $gslm_output_dir \
#     --device $device \
#     --test_only

# conduct ppl evaluation on SALMON
python src/gslm/tools/gslm_speech_ppl_wrapper.py \
    --language_model_dir $gslm_pretrained_model_dir \
    --testing_audio_fpath $data_sample_dir/61-70968-0000_orig.flac \
    --output_dir $gslm_output_dir \
    --device $device

# conduct raw unit extraction
# python src/gslm/tools/gslm_speech_ppl_wrapper.py \
#     --language_model_dir $gslm_pretrained_model_dir \
#     --testing_audio_fpath $data_sample_dir/61-70968-0000_orig.flac \
#     --output_dir $gslm_output_dir \
#     --device $device \
#     --extract_raw_units