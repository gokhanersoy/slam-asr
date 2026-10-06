import json
import copy
import torch
import numpy as np
import whisper

class SpeechDatasetJsonl(torch.utils.data.Dataset):
    def __init__(self, dataset_config, tokenizer, split="train"):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.IGNORE_INDEX = -100
        self.mel_size = getattr(dataset_config, "mel_size", 128)
        self.inference_mode = getattr(dataset_config, "inference_mode", False)

        self.prompt = getattr(
            dataset_config,
            "prompt",
            "Aşağıdaki konuşma sesini Türkçe metne dönüştürün. Sadece konuşmanın tam transcriptini yazın."
        )

        if split == "train":
            data_path = dataset_config.train_data_path
        elif split == "val" or split == "validation":
            data_path = dataset_config.val_data_path
        else:
            data_path = getattr(dataset_config, "test_data_path", dataset_config.val_data_path)

        self.data_list = []
        with open(data_path, encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    self.data_list.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict["source"]
        target = data_dict.get("target", "")
        key = data_dict.get("key", str(index))

        audio_raw = whisper.load_audio(audio_path)
        audio_raw = whisper.pad_or_trim(audio_raw)
        audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
        
        # Calculate expected audio tokens after 2x whisper downsampling & 5x linear stacking downsampling
        audio_length = (audio_mel.shape[0] + 1) // 2
        audio_length = audio_length // 5

        audio_pseudo = torch.full((audio_length,), -1)

        prompt_str = f"<|im_start|>user\n{self.prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_tensor))
            example_mask = example_ids.ge(-1)

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_mel": audio_mel,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        answer_str = f"{target}<|im_end|>\n"
        example_str = prompt_str + answer_str
        example_ids = self.tokenizer.encode(example_str, add_special_tokens=False)
        example_ids.append(self.tokenizer.eos_token_id)
        example_tensor = torch.tensor(example_ids, dtype=torch.int64)

        example_ids = torch.cat((audio_pseudo, example_tensor))

        labels_ids = copy.deepcopy(example_ids)
        labels_ids[: audio_length + prompt_length] = self.IGNORE_INDEX
        example_mask = example_ids.ge(-1)

        label_mask = labels_ids.ge(0)
        example_ids[~example_mask] = self.tokenizer.pad_token_id
        labels_ids[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio_mel": audio_mel,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
            "key": key,
            "target": target,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                return torch.cat((sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            return sequence[:max_length]
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_prompt_lengths = [s["audio_length"] + s["prompt_length"] for s in samples]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s["prompt_length"] for s in samples]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)

        input_ids = torch.stack([
            self.pad_custom(samples[idx]["input_ids"], input_prompt_lengths[idx], input_prompt_max_length, input_answer_lengths[idx], input_answer_max_length, self.tokenizer.pad_token_id)
            for idx in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.pad_custom(samples[idx]["attention_mask"], input_prompt_lengths[idx], input_prompt_max_length, input_answer_lengths[idx], input_answer_max_length, False)
            for idx in range(len(samples))
        ])

        audio_mel = torch.stack([s["audio_mel"] for s in samples])

        modality_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for idx in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[idx]
            modality_mask[idx, padding_left : padding_left + samples[idx]["audio_length"]] = True

        if self.inference_mode:
            keys = [s["key"] for s in samples]
            targets = [s["target"] for s in samples]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_mel": audio_mel,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
            }

        labels = torch.stack([
            self.pad_custom(samples[idx]["labels"], input_prompt_lengths[idx], input_prompt_max_length, input_answer_lengths[idx], input_answer_max_length, self.IGNORE_INDEX)
            for idx in range(len(samples))
        ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio_mel": audio_mel,
            "modality_mask": modality_mask,
        }

    def pad_custom(self, tensor, p_len, max_p_len, a_len, max_a_len, pad_value):
        pad_p = max_p_len - p_len
        pad_a = max_a_len - a_len
        
        prompt_part = tensor[:p_len]
        answer_part = tensor[p_len:]

        padded_prompt = torch.cat([torch.full((pad_p,), pad_value, dtype=tensor.dtype), prompt_part])
        padded_answer = torch.cat([answer_part, torch.full((pad_a,), pad_value, dtype=tensor.dtype)])

        return torch.cat([padded_prompt, padded_answer])


def get_speech_dataset(dataset_config, tokenizer, split):
    return SpeechDatasetJsonl(dataset_config, tokenizer, split)
