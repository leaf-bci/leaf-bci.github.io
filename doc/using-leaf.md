# Using LEAF with New EEG Data

This example shows how to preprocess a new EEG dataset, align its electrodes with the LEAF montage, extract EEG embeddings, and compare them with text labels. Run the code from the root of the LEAF repository.

Assume that each trial is four seconds long and was recorded from 32 channels at 250 Hz. The input array has shape `(N, 32, 1000)`, where `N` is the number of trials.

## Preprocess and align the channels

```python
import mne
import numpy as np

from leaf_datasets.shared import pipeline

channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
    'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4',
    'P8', 'POz', 'O1', 'Oz', 'O2', 'AF3', 'AF4', 'FCz',
]

# Replace this random array with your epoched EEG data.
raw_eeg = np.random.randn(2, 32, 1000).astype(np.float32)

info = mne.create_info(channels, sfreq=250, ch_types='eeg')
epochs = mne.EpochsArray(raw_eeg, info, verbose=False)
epochs.filter(l_freq=0.1, h_freq=70, method='iir', verbose=False)
epochs.resample(200, verbose=False)

eeg = pipeline(epochs.get_data().astype(np.float32), channels)
print(eeg.shape)  # (2, 65, 800)
```

`pipeline` performs three operations:

1. It clips values outside the 0.1 and 99.9 percentiles.
2. It applies robust scaling using the median and interquartile range computed over the supplied array.
3. It maps the named electrodes to the LEAF 65-channel montage and interpolates missing template channels.

Channel names must match an electrode or alias in `configs/LEAF-ch65/LEAF-ch65.json`. Add an alias there if a dataset uses a different channel name. Interpolation is intended to align established EEG montages; verify the channel names and electrode positions, and retain as much coverage of the 65-channel template as the recording provides.

## Input length and padding

The default LEAF configuration uses a sampling rate of 200 Hz and accepts up to 2,000 samples, or 10 seconds. During pretraining and instruction tuning, shorter trials are zero-padded to 10 seconds and longer trials are cropped. The padding or crop position is randomized for training samples; validation samples are padded on the right or cropped from the beginning.

Padding to 10 seconds is not required for inference. The four-second example above contains 800 samples and can be passed to LEAF directly. The tokenizer divides it into eight non-overlapping windows of 100 samples. For recordings longer than 10 seconds, crop the signal or divide it into segments before inference.

## Extract LEAF embeddings

Download the instruction-tuned checkpoint before running this step. The checkpoint and text-embedding cache must use the same text encoder; the released model below uses MPNet.

```python
import torch

from LEAF import LEAF
from init_text_embeddings import load_embeddings
from load_config import build_model_config, load_yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = build_model_config(load_yaml('configs/LEAF_mpnet.yaml'))

model = LEAF(config)
state = torch.load(
    'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt',
    map_location='cpu',
    weights_only=True,
)
model.load_state_dict(state, strict=True)
model.to(device).eval()

text_embeddings = load_embeddings(config.text_emb_model_name)
instruction = torch.tensor(
    text_embeddings['This is an emotion recognition task'],
    dtype=torch.float32,
    device=device,
)

x = torch.tensor(eeg, dtype=torch.float32, device=device)
instruction = instruction.unsqueeze(0).expand(x.shape[0], -1)

with torch.inference_mode():
    eeg_embeddings, _ = model(x, instruction)

print(eeg_embeddings.shape)  # (2, 768) with MPNet
```

The first output contains one L2-normalized EEG embedding per trial. With MPNet, its shape is `(N, 768)`. The second output contains the EEG Tower tokens.

## Compare EEG with label text

LEAF maps the EEG trials and label texts into the same normalized embedding space. Their dot product is therefore cosine similarity. This example compares every trial with the text labels `Happy` and `Sad`:

```python
labels = ['Happy', 'Sad']
prototypes = torch.tensor(
    np.stack([text_embeddings[label] for label in labels]),
    dtype=torch.float32,
    device=device,
)

similarities = eeg_embeddings @ prototypes.T
distances = 1 - similarities
predictions = [labels[index] for index in similarities.argmax(dim=1).tolist()]

print(similarities.shape)  # (2, 2)
print(distances.shape)     # (2, 2)
print(predictions)
```

Each row contains the comparison scores for one EEG trial, and each column corresponds to one candidate label. Replace the instruction and labels with the task of interest. If the required text is not already cached, add it to `configs/tasks.yaml` and run:

```bash
python init_text_embeddings.py mpnet-base
```
