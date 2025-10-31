import os
import glob
import torch
import torchaudio
import numpy as np
from taste_speech import TasteForCausalLM, TasteProcessor

TASLM_INPUT_SAMPLING_RATE = 16000
TASLM_OUTPUT_SAMPLING_RATE = 22050


class TaslmSpeechPPLWrapper:
    def __init__(
        self,
        pretrained_model_dir: str,
        attn_implementation: str = "sdpa",
        device: str = "cpu",
    ):
        self.device = device
        self.model = TasteForCausalLM.from_pretrained(
            pretrained_model_dir,
            attn_implementation=attn_implementation,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = TasteProcessor.from_pretrained(
            pretrained_model_dir
        )
        self.generator = self.processor.get_generator(device=self.device)
        self.generate_kwargs = dict(
            llm_tokenizer=self.processor.llm_tokenizer,
            asr_tokenizer=self.processor.audio_tokenizer,
            extra_words=16,
            text_top_p=0.3,
            taste_top_p=0.0,  # not activated for audio embedding continuation
            text_temperature=0.5,
            repetition_penalty=1.1,
            debug=False,
        )
    
    @torch.no_grad()
    def get_per_token_losses(
        self,
        audio_sample
    ) -> torch.Tensor:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
            sr = 16000  # assume the input audio is always 16kHz
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        if sr != TASLM_INPUT_SAMPLING_RATE:
            # resample to 16kHz
            raw_audio = torchaudio.functional.resample(
                raw_audio,
                orig_freq=sr,
                new_freq=TASLM_INPUT_SAMPLING_RATE,
            )
            sr = TASLM_INPUT_SAMPLING_RATE
        # process audio
        inputs = self.processor(
            raw_audio,
            sr,
            ref_audio_list=[raw_audio],
            output_text_info=True
        )
        inputs = inputs.to(device=self.device)
        _, llm_indices = self.model.extract_vq(
            asr_token_ids=inputs["asr_token_ids"],
            asr_token_lengths=inputs["asr_token_lengths"],
            asr_word_ids=inputs["asr_word_ids"],
            llm_token_ids=inputs["llm_token_ids"],
            llm_token_lengths=inputs["llm_token_lengths"],
            audio_features=inputs["audio_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
        )