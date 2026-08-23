import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, WhisperModel, WhisperFeatureExtractor


class Wav2Vec2StyleWrapper(nn.Module):
    def __init__(self, model_name: str, device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        self.frame_rate_hz = 49.95
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def preprocess(self, waveforms: list, sampling_rate: int = 16000):
        inputs = self.feature_extractor(
            waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        return inputs.input_values.to(self.device)

    @torch.no_grad()
    def all_hidden_states(self, input_values):
        out = self.model(input_values)
        return out.hidden_states

    def num_layers(self):
        return self.model.config.num_hidden_layers

    @torch.no_grad()
    def extract_layer(self, input_values, layer_idx: int):
        hs = self.all_hidden_states(input_values)
        return hs[layer_idx]

    def head_from_layer(self, layer_idx: int):
        encoder = self.model.encoder
        remaining_layers = encoder.layers[layer_idx:]
        is_stable = bool(getattr(self.model.config, "do_stable_layer_norm", False))
        final_ln = getattr(encoder, "layer_norm", None) if is_stable else None

        class RemainingLayers(nn.Module):
            def __init__(self, layers, final_layer_norm=None):
                super().__init__()
                self.layers = layers
                self.final_layer_norm = final_layer_norm

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)[0]
                if self.final_layer_norm is not None:
                    x = self.final_layer_norm(x)
                return x

        return RemainingLayers(remaining_layers, final_ln).to(self.device).eval()


class WhisperStyleWrapper(nn.Module):
    """
    Wraps Whisper's ENCODER only (no decoder needed for stitching).
    Confirmed call signature on Unity, transformers 5.14.1:
        forward(hidden_states, attention_mask, **kwargs) -> torch.Tensor
    """
    def __init__(self, model_name: str, device="cuda"):
        super().__init__()
        self.model_name = model_name
        full_model = WhisperModel.from_pretrained(model_name)
        self.model = full_model.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def preprocess(self, waveforms: list, sampling_rate: int = 16000):
        inputs = self.feature_extractor(
            waveforms, sampling_rate=sampling_rate, return_tensors="pt"
        )
        return inputs.input_features.to(self.device)

    @torch.no_grad()
    def all_hidden_states(self, input_features):
        out = self.model(input_features, output_hidden_states=True)
        return out.hidden_states

    def num_layers(self):
        return self.model.config.encoder_layers

    @torch.no_grad()
    def extract_layer(self, input_features, layer_idx: int):
        hs = self.all_hidden_states(input_features)
        return hs[layer_idx]

    def head_from_layer(self, layer_idx: int):
        remaining_layers = self.model.layers[layer_idx:]
        final_ln = self.model.layer_norm

        class RemainingLayers(nn.Module):
            def __init__(self, layers, final_layer_norm):
                super().__init__()
                self.layers = layers
                self.final_layer_norm = final_layer_norm

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x, None)
                x = self.final_layer_norm(x)
                return x

        return RemainingLayers(remaining_layers, final_ln).to(self.device).eval()


MODEL_REGISTRY = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "hubert-base": "facebook/hubert-base-ls960",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
}

WHISPER_MODEL_REGISTRY = {
    "whisper-base": "openai/whisper-base",
}

CLAP_MODEL_REGISTRY = {
    "clap-htsat-unfused": "laion/clap-htsat-unfused",
}


class ClapStyleWrapper(nn.Module):
    """
    Wraps CLAP's AUDIO tower only (no text tower needed for stitching).
    Phase B -- SCOPED integration, deliberately not fully generalized.

    CLAP's audio backbone (HTSAT) is a Swin Transformer, not a flat stack
    of identical layers:
      - Only 4 "layers" in our sense (Swin STAGES, not individual blocks) --
        config.depths = [2, 2, 6, 2] blocks per stage, but only one hidden
        state per STAGE via output_hidden_states=True.
      - Channel dimension CHANGES between stages (patch-merging).
      - Native hidden-state shape is (B, C, H, W) -- flattened here to
        (B, H*W, C) so the rest of the pipeline can treat it like any other
        (B, T, C) tensor. This discards 2D spatial structure -- a real,
        acknowledged limitation of this scoped-down integration.
      - Sequence length shrinks stage-to-stage, so run_phase_a_sanity.py now
        includes a resampling step to handle mismatched T between layers.

    DELIBERATELY NOT IMPLEMENTED: head_from_layer(). The self-stitch check
    never calls it. FFM/TLT will NOT work with this wrapper yet.
    """
    def __init__(self, model_name: str, device="cuda"):
        super().__init__()
        self.model_name = model_name
        from transformers import ClapModel, ClapFeatureExtractor
        full_model = ClapModel.from_pretrained(model_name)
        self.model = full_model.audio_model
        self.feature_extractor = ClapFeatureExtractor.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def preprocess(self, waveforms: list, sampling_rate: int = 16000):
        import librosa
        CLAP_SAMPLING_RATE = 48000
        if sampling_rate != CLAP_SAMPLING_RATE:
            waveforms = [
                librosa.resample(w, orig_sr=sampling_rate, target_sr=CLAP_SAMPLING_RATE)
                for w in waveforms
            ]
        inputs = self.feature_extractor(
            waveforms, sampling_rate=CLAP_SAMPLING_RATE, return_tensors="pt"
        )
        return inputs["input_features"].to(self.device)

    @torch.no_grad()
    def all_hidden_states(self, input_features):
        out = self.model(input_features, output_hidden_states=True)
        flattened = tuple(h.flatten(2).transpose(1, 2) for h in out.hidden_states)
        return flattened

    def num_layers(self):
        return len(self.model.config.depths)

    @torch.no_grad()
    def extract_layer(self, input_features, layer_idx: int):
        hs = self.all_hidden_states(input_features)
        return hs[layer_idx]

    def head_from_layer(self, layer_idx: int):
        raise NotImplementedError(
            "ClapStyleWrapper does not implement head_from_layer -- this scoped "
            "integration only supports the label-free self-stitch check (LFM), "
            "which never calls this. FFM/TLT need this and will not work with "
            "CLAP yet; it would require handling Swin's patch-merging "
            "downsampling between stages explicitly."
        )


AUDIOMAE_MODEL_REGISTRY = {
    "audiomae": "hf-hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
}


class AudioMAEStyleWrapper(nn.Module):
    """
    Wraps the AudioMAE ViT-B/16 encoder -- a community `timm` port of the
    original Facebook AudioMAE weights.
    """
    def __init__(self, model_name: str, device="cuda"):
        super().__init__()
        import timm
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval().to(device)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sample_rate = 16000
        self.n_mels = 128
        self.win_length = int(0.025 * self.sample_rate)
        self.hop_length = int(0.010 * self.sample_rate)
        self.target_frames = 1024

    @torch.no_grad()
    def preprocess(self, waveforms: list, sampling_rate: int = 16000):
        import librosa
        import numpy as np

        specs = []
        for w in waveforms:
            if sampling_rate != self.sample_rate:
                w = librosa.resample(w, orig_sr=sampling_rate, target_sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(
                y=w, sr=self.sample_rate, n_mels=self.n_mels,
                win_length=self.win_length, hop_length=self.hop_length,
                window="hann",
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = mel_db.T
            if mel_db.shape[0] < self.target_frames:
                pad = self.target_frames - mel_db.shape[0]
                mel_db = np.pad(mel_db, ((0, pad), (0, 0)), mode="constant",
                                 constant_values=mel_db.min())
            else:
                mel_db = mel_db[:self.target_frames]
            specs.append(mel_db)

        batch = np.stack(specs)
        batch_t = torch.from_numpy(batch).float().unsqueeze(1)
        return batch_t.to(self.device)

    @torch.no_grad()
    def all_hidden_states(self, input_features):
        x = self.model.patch_embed(input_features)
        if getattr(self.model, "cls_token", None) is not None:
            cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if getattr(self.model, "pos_embed", None) is not None:
            x = x + self.model.pos_embed
        if hasattr(self.model, "pos_drop"):
            x = self.model.pos_drop(x)

        hidden_states = [x]
        for blk in self.model.blocks:
            x = blk(x)
            hidden_states.append(x)
        return tuple(hidden_states)

    def num_layers(self):
        return len(self.model.blocks)

    @torch.no_grad()
    def extract_layer(self, input_features, layer_idx: int):
        hs = self.all_hidden_states(input_features)
        return hs[layer_idx]

    def head_from_layer(self, layer_idx: int):
        raise NotImplementedError(
            "AudioMAEStyleWrapper does not implement head_from_layer -- scoped "
            "to the self-stitch check (LFM) only, same as ClapStyleWrapper."
        )


PATCHTST_MODEL_REGISTRY = {
    "patchtst": "namctin/patchtst_etth1_pretrain",
}


class PatchTSTStyleWrapper(nn.Module):
    """
    Wraps PatchTST's plain encoder (PatchTSTModel, not the forecasting-head
    variant) -- Phase D, Step 1, first non-audio model in this project.

    Verified before implementation: a genuine pretrained checkpoint exists
    (unlike the "getting started" tutorial, which trains from scratch) --
    "namctin/patchtst_etth1_pretrain" -- loaded via the standard
    PatchTSTModel.from_pretrained(...) pattern. Input is a `past_values`
    tensor (B, context_length, num_input_channels), not raw waveforms.

    SCOPE NOTE, deliberately simplified given time constraints: uses
    SYNTHETIC time-series data (generated in the calling script, not a
    real sourced dataset). Replicates the same synthetic series across
    all num_input_channels (reasonable given the checkpoint's
    channel-independent architecture).

    UNVERIFIED RISK: output_hidden_states support/shape not yet directly
    confirmed against this checkpoint. The calling script prints that
    info immediately and checks it before proceeding, combined into one
    run rather than a separate diagnostic job, given the time constraint.

    SCOPED, like ClapStyleWrapper/AudioMAEStyleWrapper: self-stitch
    check (LFM) only. head_from_layer is not implemented.
    """
    def __init__(self, model_name: str, device="cuda"):
        super().__init__()
        from transformers import PatchTSTModel
        self.model_name = model_name
        self.model = PatchTSTModel.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.context_length = self.model.config.context_length
        self.num_input_channels = self.model.config.num_input_channels

    @torch.no_grad()
    def preprocess(self, series_list: list, sampling_rate=None):
        import numpy as np
        batch = []
        for s in series_list:
            if len(s) < self.context_length:
                s = np.pad(s, (0, self.context_length - len(s)))
            else:
                s = s[:self.context_length]
            batch.append(np.tile(s[:, None], (1, self.num_input_channels)))
        batch = np.stack(batch)
        return torch.from_numpy(batch).float().to(self.device)

    @torch.no_grad()
    def all_hidden_states(self, past_values):
        out = self.model(past_values=past_values, output_hidden_states=True)
        # Confirmed via direct diagnostic run: each hidden state is
        # (B, num_input_channels, num_patches, d_model) -- 4D, not the
        # (B, T, C) every other wrapper returns, reflecting PatchTST's
        # channel-independent design (each channel gets its own separate
        # patch sequence, processed through the same shared weights).
        # Since preprocess() deliberately replicates the SAME synthetic
        # series across all channels, their outputs are redundant -- take
        # channel 0 only, rather than wastefully flattening 7 identical
        # copies together.
        return tuple(h[:, 0, :, :] for h in out.hidden_states)

    def num_layers(self):
        # Confirmed via direct diagnostic run: output_hidden_states gives
        # exactly num_hidden_layers states, with NO separate pre-transformer
        # embedding state at index 0 (unlike every other wrapper here) --
        # so len(all_hidden_states()) == num_layers(), not num_layers()+1.
        return self.model.config.num_hidden_layers

    @torch.no_grad()
    def extract_layer(self, past_values, layer_idx: int):
        hs = self.all_hidden_states(past_values)
        return hs[layer_idx]

    def head_from_layer(self, layer_idx: int):
        raise NotImplementedError(
            "PatchTSTStyleWrapper does not implement head_from_layer -- scoped "
            "to the self-stitch check (LFM) only, same as ClapStyleWrapper/"
            "AudioMAEStyleWrapper."
        )


def load_model(key: str, device="cuda"):
    if key in MODEL_REGISTRY:
        return Wav2Vec2StyleWrapper(MODEL_REGISTRY[key], device=device)
    if key in WHISPER_MODEL_REGISTRY:
        return WhisperStyleWrapper(WHISPER_MODEL_REGISTRY[key], device=device)
    if key in CLAP_MODEL_REGISTRY:
        return ClapStyleWrapper(CLAP_MODEL_REGISTRY[key], device=device)
    if key in AUDIOMAE_MODEL_REGISTRY:
        return AudioMAEStyleWrapper(AUDIOMAE_MODEL_REGISTRY[key], device=device)
    if key in PATCHTST_MODEL_REGISTRY:
        return PatchTSTStyleWrapper(PATCHTST_MODEL_REGISTRY[key], device=device)
    raise ValueError(
        f"Unknown model key '{key}'. Options: "
        f"{list(MODEL_REGISTRY) + list(WHISPER_MODEL_REGISTRY) + list(CLAP_MODEL_REGISTRY) + list(AUDIOMAE_MODEL_REGISTRY) + list(PATCHTST_MODEL_REGISTRY)}"
    )
