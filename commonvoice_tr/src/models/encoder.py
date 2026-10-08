import types
import torch
import torch.nn as nn
import torch.nn.functional as F

class WhisperWrappedEncoder:
    @classmethod
    def load(cls, model_config):
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            target_dtype = self.conv1.weight.dtype
            x = x.to(dtype=target_dtype)
            
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        encoder_path = getattr(model_config, "encoder_path", "openai/whisper-large-v3")
        encoder_path_hf = getattr(model_config, "encoder_path_hf", None)

        if encoder_path_hf is not None:
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(encoder_path_hf, torch_dtype=torch.float32).encoder
        else:
            import whisper
            model_name = encoder_path.replace("openai/whisper-", "").replace("openai/", "")
            whisper_model = whisper.load_model(name=model_name, device='cpu')
            encoder = whisper_model.encoder
            encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
            
        return encoder
