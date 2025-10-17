
# GSLM
## Setup Environment
```bash
git clone git@github.com:facebookresearch/textlesslib.git
cd textlesslib
uv python install 3.9
uv venv --python 3.9
# download older version of torch first
uv pip install torch==1.13.1 torchaudio==0.13.1 transformers==4.53.0 datasets==3.6.0
# install fairseq mannually
uv pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8
uv pip install -e .
```

## Prepare GSLM Pretrained Model
```bash
mkdir -p ./work/pretrained_models/GSLM
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km100/hubert100_lm.tgz -O ./work/pretrained_models/GSLM/hubert100_lm.tgz
cd ./work/pretrained_models/GSLM
tar -xzvf hubert100_lm.tgz
cd ../../..
```

## Prepare Your Own Audio Sample for Testing

## Test Model Continuation
```bash
# under speech_ppl
# NOTE: you may need to modify the $root_dir in the below script
bash ./src/gslm/scripts/test_gslm_continuation.sh
```

## Speech PPL (greedy) on salmon
```bash
bash ./src/gslm/scripts/generate_greedy_result.sh
```


# TWIST

## Setup Environment
See [GSLM](#gslm)

## Prepare TWIST Pretrained Model
```bash
mkdir -p ./work/pretrained_models/twist
cd ./work/pretrained_models/twist
wget https://dl.fbaipublicfiles.com/textless_nlp/twist/lms/TWIST-1.3B.zip
mkdir TWIST-1.3B
unzip TWIST-1.3B.zip -d TWIST-1.3B
cd ../../..
```