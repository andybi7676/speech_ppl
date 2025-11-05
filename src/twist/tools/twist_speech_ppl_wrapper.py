import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from transformers import AutoModelForCausalLM
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder
from textless.data.speech_encoder import SpeechEncoder

MODEL_NAME="TWIST1.3B"

class TwistSpeechPPLWrapper:
    def __init__(
        self,
        twist_model_pretrained_path,
        dense_model="mhubert-base-25hz",
        quantizer_model="kmeans",
        vocab=500,
        device=None,
    ):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load speech encoder and vocoder
        self.encoder = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer_model,
            vocab_size=vocab,
            deduplicate=False, # set to False but mannually deduplicate later if needed
            need_f0=False,
            add_bos_eos=False,
        ).eval().to(self.device)

        self.vocoder = CodeHiFiGANVocoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer_model,
            vocab_size=vocab
        ).eval().to(self.device)

        # build twist unit lm
        self.twist_lm = AutoModelForCausalLM.from_pretrained(twist_model_pretrained_path).to(self.device)
        self.twist_lm.eval()
        self.output_sample_rate = self.vocoder.output_sample_rate
    
    @torch.no_grad()
    def extract_raw_units(
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
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        # get audio units
        encoder_output = self.encoder(raw_audio)
        units = encoder_output['units']
        # print(units.shape)
        # print(raw_audio.shape)
        # assert False, "Debug extract units"
        return units.cpu()

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
        # get input ids for unit lm
        units = self.encoder(raw_audio)['units']
        # perform deduplication
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True)
        input_ids = input_ids.unsqueeze(0) + self.twist_lm.config.offset # add batch dim (1, seq_len)
        # prepare labels
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit lm logits
        logits = self.twist_lm(input_ids)[0]
        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )
        # return loss_all_tokens
        return {
            'loss_per_token': loss_all_tokens,
            'raw_units': units
        }

    @torch.no_grad()
    def generate_continuation_audio(
        self,
        audio_sample,
        offset=None
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
        # get input ids for unit lm
        units = self.encoder(raw_audio)['units']
        # perform deduplication
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True)
        input_ids = input_ids.unsqueeze(0)  # add batch dim (1, seq_len)
        # generate continuation ids
        if offset is None:
            offset = self.twist_lm.config.offset
            input_len= int(input_ids.shape[-1])
            generation_len = int(min(250, 5 * input_len))
        generated_ids = self.twist_lm.generate(offset + input_ids, max_length=generation_len, min_length=3*input_len, do_sample=True, temperature=0.8) - offset
        full_generation = self.vocoder(generated_ids, dur_prediction=True)

        return full_generation.cpu()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Twist Speech PPL Wrapper Test")
    argparser.add_argument("--twist_model_pretrained_path", type=str, required=True, help="Path to pretrained twist model")
    argparser.add_argument("--input_audio_fpath", type=str, required=True, help="Path to testing input audio file")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    argparser.add_argument("--prompt_duration_sec", type=float, default=None, help="Duration of the prompt in seconds")
    argparser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cpu' or 'cuda'")
    argparser.add_argument("--test_only", action="store_true", help="Only test per token loss and generation")
    argparser.add_argument("--extract_main_ppl_results", action="store_true")
    argparser.add_argument("--extract_raw_units", action="store_true")
    argparser.add_argument("--extract_additional_ppl_results", action="store_true")
    argparser.add_argument("--skip_generation", action="store_true", help="Skip audio generation")
    argparser.add_argument("--overwrite_model_name", type=str, default=None, help="Overwrite model name for saving results" )
    args = argparser.parse_args()
    if args.overwrite_model_name:
        MODEL_NAME = args.overwrite_model_name
    os.makedirs(args.output_dir, exist_ok=True)
    # get device
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # build model
    model = TwistSpeechPPLWrapper(
        twist_model_pretrained_path=args.twist_model_pretrained_path,
        device=device
    )
    print("Model Output Sample Rate:", model.output_sample_rate)
    if args.test_only:
        # load audio for testing
        audio, sr = torchaudio.load(args.input_audio_fpath)
        audio = audio.to(device)
        # get per token losses
        per_token_losses = model.get_per_token_losses(audio)
        print("Per token losses:", per_token_losses)
        # try conditional generation
        # optionally trim audio
        if args.prompt_duration_sec:
            prompt = int(args.prompt_duration_sec * sr)
            audio = audio[:, :prompt]
        generated_audio = model.generate_continuation_audio(audio).cpu().unsqueeze(0)
        print("Generated audio shape:", generated_audio.shape)
        # save generated audio to output dir
        input_fid = os.path.basename(args.input_audio_fpath).split('.')[0]
        output_audio_fpath = os.path.join(args.output_dir, f"{input_fid}_contd_gen.wav")
        prompt_audio_fpath =  os.path.join(args.output_dir, f"{input_fid}_prompt.wav")
        torchaudio.save(prompt_audio_fpath, audio.cpu(), sr)
        torchaudio.save(output_audio_fpath, generated_audio, model.output_sample_rate)
        exit(0)
    
    from datasets import Audio, load_dataset, load_from_disk
    if args.extract_main_ppl_results:
        # start SALMON dataset test
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
            positive_sample_loss_results = get_per_token_losses(e["positive_audio"])
            e["postive_sample_tokenwise_loss"] = positive_sample_loss_results['loss_per_token']
            e["postive_sample_raw_units"] = positive_sample_loss_results['raw_units']
            negative_sample_loss_results = get_per_token_losses(e["negative_audio"])
            e["negative_sample_tokenwise_loss"] = negative_sample_loss_results['loss_per_token']
            e["negative_sample_raw_units"] = negative_sample_loss_results['raw_units']
            if "consistency" in e["task"]:
                prompt_sample_loss_results = get_per_token_losses(e["prompt_audio"])
                e["prompt_sample_tokenwise_loss"] = prompt_sample_loss_results['loss_per_token']
                e["prompt_sample_raw_units"] = prompt_sample_loss_results['raw_units']
                positive_continuation_result = get_per_token_losses(e["continuation_audio_positive"])
                negative_continuation_result = get_per_token_losses(e["continuation_audio_negative"])
                e["positive_continuation_tokenwise_loss"] = positive_continuation_result['loss_per_token']
                e["positive_continuation_raw_units"] = positive_continuation_result['raw_units']
                e["negative_continuation_tokenwise_loss"] = negative_continuation_result['loss_per_token']
                e["negative_continuation_raw_units"] = negative_continuation_result['raw_units']
                if not args.skip_generation:
                    generated_audio = generate_continuation_audio(e["prompt_audio"])
                    e["model_generated_continuation"] = {"sampling_rate": model.output_sample_rate, "array": generated_audio.squeeze().numpy()}
            
            e["code_frame_rate"] =  25,
            e["code_depth"] =  1
            e["offset"] = model.twist_lm.config.offset,
            e["model_sampling_rate"] = model.output_sample_rate,
            e["ppl_sanity"] = int((e["postive_sample_tokenwise_loss"].mean() < e["negative_sample_tokenwise_loss"].mean()).item()) #sanity check if number same as SALMon, would be rerun with other methods
            # print("sample correct:",e["ppl_sanity"])
            return e
        # try hf salmon dataset
        splts = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splts = ['bg_alignment', 'sentiment_alignment']
        # splts = ['gender_consistency', 'speaker_consistency']
        # splts = ['bg_all_consistency']
        for splt in splts:
            print(splt)
            ds = load_dataset("SpeechPPL/SALMon_with_meta", splt)
            # ds["train"] = ds["train"].select([0])  # for testing purpose, only use 5 samples
            ds = ds.map(get_model_features)
            if not args.skip_generation:
                ds = ds.cast_column("model_generated_continuation", Audio(sampling_rate=model.output_sample_rate))
            # save results
            save_dir = os.path.join(args.output_dir, MODEL_NAME, splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print("Results saved to", save_dir)
            if not args.skip_generation:
                # save 10 prompt contd generation results
                gen_save_dir = os.path.join(save_dir, "generation_examples")
                os.makedirs(gen_save_dir, exist_ok=True)
                for i, item in enumerate(ds['train'].select([0, 1, 2, 3, 4, 5])):
                    if "consistency" in item["task"]:
                        gen_audio = item["model_generated_continuation"]["array"]
                        gen_sr = item["model_generated_continuation"]["sampling_rate"]
                        gen_fpath = os.path.join(gen_save_dir, f"{i}_gen.wav")
                        torchaudio.save(gen_fpath, torch.Tensor(gen_audio).unsqueeze(0), gen_sr)
                        prompt_audio = item["prompt_audio"]["array"]
                        prompt_sr = 16000
                        prompt_fpath = os.path.join(gen_save_dir, f"{i}_prompt.wav")
                        torchaudio.save(prompt_fpath, torch.Tensor(prompt_audio).unsqueeze(0), prompt_sr)
                print("Generation examples saved to", gen_save_dir)
            # push to hub
            # calculate ppl sanity
            ppl_sanity = sum(ds['train']['ppl_sanity'])
            print(f"Accuracy on {splt}: {ppl_sanity} / {len(ds['train'])} = {ppl_sanity / len(ds['train']):.4f}")
            ds.push_to_hub(f"SpeechPPL/SALMon_{MODEL_NAME}", config_name=splt)
    if args.extract_raw_units:
        print(
            "Currently extracting raw units has been merged to the main ppl results extraction. \
            Please use --extract_main_ppl_results, which will also extract raw units."
        )
    if args.extract_additional_ppl_results:
        print(
            "Currently extracting additional ppl results has been merged to the main ppl results extraction. \
            Please use --extract_main_ppl_results, which will also extract additional ppl results."
        )