import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.models.encoder import WhisperWrappedEncoder
from src.models.projector import EncoderProjectorConcat

def setup_tokenizer(train_config, model_config, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def setup_encoder(train_config, model_config, **kwargs):
    encoder = WhisperWrappedEncoder.load(model_config)
    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
    return encoder

def setup_llm(train_config, model_config, **kwargs):
    quantization = getattr(train_config, "quantization", False)
    use_fp16 = getattr(train_config, "use_fp16", True)
    torch_dtype = torch.float16 if use_fp16 else torch.bfloat16

    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        llm = AutoModelForCausalLM.from_pretrained(
            model_config.llm_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            model_config.llm_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    if hasattr(llm, "config"):
        llm.config.use_cache = False

    if train_config.freeze_llm:
        for name, param in llm.named_parameters():
            param.requires_grad = False
        llm.eval()

    if train_config.use_peft:
        peft_cfg = kwargs.get("peft_config", None)
        if peft_cfg is not None:
            target_modules = getattr(peft_cfg, "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
            lora_config = LoraConfig(
                r=getattr(peft_cfg, "r", 16),
                lora_alpha=getattr(peft_cfg, "lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=getattr(peft_cfg, "lora_dropout", 0.05),
                bias=getattr(peft_cfg, "bias", "none"),
                task_type="CAUSAL_LM",
            )
            llm = get_peft_model(llm, lora_config)
            if hasattr(llm, "gradient_checkpointing_enable"):
                llm.gradient_checkpointing_enable()
            llm.print_trainable_parameters()

    return llm

def setup_encoder_projector(train_config, model_config, **kwargs):
    return EncoderProjectorConcat(model_config)

def model_factory(train_config, model_config, **kwargs):
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    encoder = setup_encoder(train_config, model_config, **kwargs)
    llm = setup_llm(train_config, model_config, **kwargs)
    encoder_projector = setup_encoder_projector(train_config, model_config, **kwargs)

    model = SLAMASRModel(
        encoder=encoder,
        llm=llm,
        encoder_projector=encoder_projector,
        tokenizer=tokenizer,
        train_config=train_config,
        model_config=model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    return model, tokenizer


class SLAMASRModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.IGNORE_INDEX = -100

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if audio_mel is not None:
            device = audio_mel.device
            if self.encoder is not None:
                self.encoder.to(device)
            if self.encoder_projector is not None:
                self.encoder_projector.to(device)

            if self.train_config.freeze_encoder:
                self.encoder.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False):
                        audio_input = audio_mel.permute(0, 2, 1).to(device=device, dtype=torch.float32)
                        encoder_outs = self.encoder.extract_variable_length_features(audio_input)
            else:
                audio_input = audio_mel.permute(0, 2, 1).to(device=device)
                encoder_outs = self.encoder.extract_variable_length_features(audio_input)

            encoder_outs = self.encoder_projector(encoder_outs.float())

        input_ids_clean = input_ids.clone()
        input_ids_clean[input_ids_clean < 0] = 0

        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(input_ids_clean)
        elif hasattr(self.llm, "get_input_embeddings"):
            inputs_embeds = self.llm.get_input_embeddings()(input_ids_clean)
        else:
            inputs_embeds = self.llm.model.model.embed_tokens(input_ids_clean)

        if modality_mask is not None and audio_mel is not None:
            encoder_outs = encoder_outs.to(inputs_embeds.dtype)
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                length = min(modality_lengths[i], encoder_outs.shape[1])
                start_idx = modality_mask_start_indices[i]
                encoder_outs_pad[i, start_idx:start_idx + length] = encoder_outs[i, :length]

            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

    @torch.no_grad()
    def generate_transcription(
        self,
        audio_mel: torch.Tensor,
        prompt: str = "Aşağıdaki konuşma sesini Türkçe metne dönüştürün. Sadece konuşmanın tam transcriptini yazın.",
        max_new_tokens: int = 128,
        **kwargs,
    ):
        device = audio_mel.device
        if self.encoder is not None:
            self.encoder.to(device)
        if self.encoder_projector is not None:
            self.encoder_projector.to(device)

        with torch.amp.autocast('cuda', enabled=False):
            audio_input = audio_mel.permute(0, 2, 1).to(device=device, dtype=torch.float32)
            encoder_outs = self.encoder.extract_variable_length_features(audio_input)
        encoder_outs = self.encoder_projector(encoder_outs.float())

        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

        if hasattr(self.llm, "get_input_embeddings"):
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        else:
            prompt_embeds = self.llm.model.embed_tokens(prompt_ids)

        encoder_outs = encoder_outs.to(prompt_embeds.dtype)
        inputs_embeds = torch.cat((encoder_outs, prompt_embeds), dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return decoded_text.strip()
