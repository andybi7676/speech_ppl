import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import logging
from omegaconf import OmegaConf
from fairseq import utils
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.tacotron2.vocoder import TacotronVocoder
from sampler import UnitLanguageModelSampler

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME="GSLM-greedy"
GSLM_SAMPLING_RATE = 16000

class GslmSpeechPplWrapper:
    def __init__(
        self, 
        language_model_data_dir: str,
        seed: int = None,
        temperature: float = 0.7,
        vocab_size: int = 100,
        device: str = "cpu",
    ):
        logger.info("Initializing the GSLM pipeline.")
        self.device = torch.device("cuda")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            utils.set_torch_seed(seed)
        self.sampling_rate = GSLM_SAMPLING_RATE
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.tokens_framerate = 0.02  # HuBERT framerate
        self.max_length = 1000
        self.trim_trailing_audio_frames = 200
        self.sampling_kwargs = {
            "temperature": self.temperature,
            "sampling": True,
            "beam": 1,
            "prefix_size": -1,
            "max_len_a": 0.0,
            "max_len_b": self.max_length,
        }
        logger.info("... Loading the language model")
        self.sampler = UnitLanguageModelSampler.from_pretrained(
            language_model_data_dir,
        )
        logger.info("=> Done!")
        logger.info("... Loading the encoder")

        self.speech_encoder = SpeechEncoder.by_name(
            dense_model_name="hubert-base-ls960",
            quantizer_model_name="kmeans",
            vocab_size=vocab_size,
            need_f0=False,
            deduplicate=True,
            f0_normalizer=None,
            f0_quantizer=None,
        )

        logger.info("=> Done!")
        logger.info("... Loading the vocoder")
        self.resynthesizer = TacotronVocoder.by_name(
            dense_model_name="hubert-base-ls960",
            quantizer_model_name="kmeans",
            vocab_size=vocab_size,
        )
        # move to device and eval mode
        self.device = device
        self.speech_encoder = self.speech_encoder.to(self.device)
        # self.sampler.model = self.sampler.model.to(self.device)
        self.sampler = self.sampler.to(self.device) # make sure the sampler knows the device
        logger.info(f"Sampler model device: {self.sampler.device}")
        self.resynthesizer = self.resynthesizer.to(self.device)
        self.speech_encoder.eval()
        self.sampler.model.eval()
        self.resynthesizer.eval()

        logger.info("=> Done!")
        logger.info("GSLM pipeline initialized!")

    @torch.no_grad()
    def get_per_token_losses(
        self,
        audio_sample
    ) -> torch.Tensor:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        # get audio units
        encoder_output = self.speech_encoder(raw_audio)
        input_ids = encoder_output['units'].unsqueeze(0)  # (1, seq_len)

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit lm logits
        logits = self.sampler.model(input_ids)[0] # skip special tokens
        logits = logits[..., :self.vocab_size] 
        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )
        return loss_all_tokens
    
    @torch.no_grad()
    def generate_continuation_audio(
        self,
        audio_sample,
    ):
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        sample = self.speech_encoder(raw_audio)
        units = sample["units"]
        duration = sample["durations"].sum().item()
        prefix_duration = self.tokens_framerate * duration
        target_duration = self.tokens_framerate * (
            self.max_length - self.trim_trailing_audio_frames
        )

        unit_str = " ".join(list(map(str, units.tolist())))
        sampled_unit_str = self.sampler.sample([unit_str], **self.sampling_kwargs)[0]

        audio = self.resynthesizer(sampled_unit_str)
        audio = audio[
            : int(
                self.resynthesizer.output_sample_rate
                * (prefix_duration + target_duration)
            )
        ]
        return audio.cpu()

if __name__ == "__main__":
    testing_audio_fpath = "./work/data/samples/61-70968-0000_orig.flac"
    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = GslmSpeechPplWrapper(
        language_model_data_dir="./work/pretrained_models/gslm/hubert100_lm",
        seed=None,
        temperature=0.7,
        vocab_size=100,
        device=device,
    )

    # function for localizing ppl function
    def get_per_token_losses(
        audio_sample: torch.Tensor,
    ) -> torch.Tensor:
        return model.get_per_token_losses(audio_sample)
    # function for localizing generation function
    def generate_continuation_audio(
        prompt_audio: torch.Tensor,
    ) -> torch.Tensor:
        return model.generate_continuation_audio(prompt_audio)
    # function for localizing get_model_features function

    def get_model_features(e):
        # audio is 16000hz, maybe resample 
        e["postive_sample_tokenwise_loss"] = get_per_token_losses(e["positive_audio"])
        e["negative_sample_tokenwise_loss"] = get_per_token_losses(e["negative_audio"])
        if "consistency" in e["task"]:
            e["prompt_sample_tokenwise_loss"] = get_per_token_losses(e["prompt_audio"])
            generated_audio = generate_continuation_audio(e["prompt_audio"])
            e["model_generated_continuation"] = {"sampling_rate": model.sampling_rate, "array": generated_audio.squeeze().numpy()}
        
        
        e["code_frame_rate"] =  12,
        e["code_depth"] =  4
        e["model_sampling_rate"] = model.sampling_rate,
        e["ppl_sanity"] = int((e["postive_sample_tokenwise_loss"].mean() < e["negative_sample_tokenwise_loss"].mean()).item()) #sanity check if number same as SALMon, would be rerun with other methods
        print("sample correct:",e["ppl_sanity"])
        return e

    # load audio for testing
    # audio, sr = torchaudio.load(testing_audio_fpath)
    # audio = audio.to(device)
    # # get per token losses
    # per_token_losses = get_per_token_losses(audio)
    # try hf salmon dataset
    from datasets import Audio, load_dataset
    splts = ['bg_all_consistency', 'bg_domain_consistency']
    for splt in splts:
        print(splt)
        ds = load_dataset("SpeechPPL/SALMon_with_meta", splt)
        #ds["train"] = ds["train"].select([1])
        ds = ds.map(get_model_features)
        ds = ds.cast_column("model_generated_continuation", Audio(sampling_rate=model.sampling_rate))
        # print(ds[0])
        # save results
        save_dir = f"./work/outputs/gslm/speech_ppl/{MODEL_NAME}/{splt}"
        os.makedirs(save_dir, exist_ok=True)
        ds.save_to_disk(save_dir)
        print("Results saved to", save_dir)
        # push to hub
        ds.push_to_hub(f"SpeechPPL/SALMon_{MODEL_NAME}", config_name=splt)