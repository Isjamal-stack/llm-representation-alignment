"""
Shared, hardened utilities for loading audio and extracting model
activations -- used by every stitching script.
"""
import io
import torch
import soundfile as sf
from datasets import load_dataset, Audio


def _load_dataset_slice(dataset_name, dataset_config, split, num_examples):
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(min(num_examples, len(ds))))
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def _decode_waveform(example, max_samples):
    audio = example["audio"]
    if audio.get("bytes") is not None:
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
    else:
        data, sr = sf.read(audio["path"])
    if len(data) > max_samples:
        data = data[:max_samples]
    return data


def get_audio_batch(num_examples: int, sampling_rate: int = 16000, max_samples: int = 160000,
                     dataset_name: str = "hf-internal-testing/librispeech_asr_dummy",
                     dataset_config: str = "clean", split: str = "validation"):
    """Pulls a small slice of a public speech dataset, decoded manually via
    soundfile (avoids the torchcodec dependency newer `datasets` versions
    want for their built-in decode path), with each clip truncated to
    `max_samples` (default 160000 = 10s @ 16kHz) so one long outlier can't
    blow up the whole batch's padded length."""
    ds = _load_dataset_slice(dataset_name, dataset_config, split, num_examples)
    return [_decode_waveform(ex, max_samples) for ex in ds]


def get_labeled_audio_batch(num_examples: int, label_column: str = "speaker_id",
                             sampling_rate: int = 16000, max_samples: int = 160000,
                             dataset_name: str = "hf-internal-testing/librispeech_asr_dummy",
                             dataset_config: str = "clean", split: str = "validation"):
    """
    Like get_audio_batch, but also returns integer-encoded labels from
    `label_column`. Needed for TLT/FFM (unlike LFM/CKA, which are
    label-free), since Task Loss Training requires an actual downstream
    classification target.

    SCOPE NOTE: this defaults to speaker-ID classification on the same
    LibriSpeech dummy split used elsewhere in Phase A. That split is tiny
    and was never meant as a labeled benchmark -- treat results from it as
    "does the TLT/FFM code path work correctly", not as a real downstream-
    task accuracy number. Swap dataset_name/dataset_config/split/
    label_column for a proper task (e.g. a SUPERB benchmark) before
    reporting results as a genuine finding.

    Returns: (waveforms, labels, label2idx)
      waveforms: list of numpy arrays
      labels: list of int, same length, indices into label2idx
      label2idx: dict mapping the raw label values seen -> integer index
    """
    ds = _load_dataset_slice(dataset_name, dataset_config, split, num_examples)

    waveforms, raw_labels = [], []
    for ex in ds:
        waveforms.append(_decode_waveform(ex, max_samples))
        raw_labels.append(ex[label_column])

    unique_labels = sorted(set(raw_labels), key=str)
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels = [label2idx[l] for l in raw_labels]

    if len(unique_labels) < 2:
        raise ValueError(
            f"Only {len(unique_labels)} unique value(s) found in label column "
            f"'{label_column}' among the first {num_examples} examples -- need "
            f"at least 2 classes for a classification task. Try increasing "
            f"--num_examples, or pick a different --label_column/--dataset_name."
        )

    return waveforms, labels, label2idx


def extract_hidden_states_chunked(model, input_values, chunk_size=8):
    """
    Runs the forward pass in small sub-batches to bound PEAK GPU memory,
    moving each chunk's hidden states to CPU immediately after computing
    them. Returns a tuple of CPU tensors, one per layer, each shaped
    (num_examples, T, C).
    """
    num_examples = input_values.shape[0]
    per_layer_chunks = None
    for start in range(0, num_examples, chunk_size):
        chunk = input_values[start:start + chunk_size]
        with torch.no_grad():
            hs = model.all_hidden_states(chunk)
        hs = [h.detach().cpu() for h in hs]
        if per_layer_chunks is None:
            per_layer_chunks = [[h] for h in hs]
        else:
            for idx, h in enumerate(hs):
                per_layer_chunks[idx].append(h)
        del hs
        torch.cuda.empty_cache()
    return tuple(torch.cat(chunks, dim=0) for chunks in per_layer_chunks)
