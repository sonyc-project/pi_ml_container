# SONYC ML Pipeline
Another version of this exists on `sonycnode` which will be used in production.

## Input data

Two modes of data input that we're considering:
 - write wav, ml reads from file on creation
    - pros:
        - simplest solution (already implemented)
        - audio params (sr, channels) contained in file
    - cons:
        - latency - need to write then read wav file
 - call embedding function in a separate process
 - pass audio buffers over sockets
    - pros:
        - no disk operations - supposedly more instantaneous
    - cons:
        - need to receive audio parameters like sample rate and # of channels


**Current plan of action**: use filesystem watching, progress to alternatives if it becomes a problem.

## Input Features
To keep the ML code simple and general, I propose that sonycnode not contain any code definitions for feature extraction. This is because feature extraction steps are dependent on the model and we want the model to be easily switchable without risking bad input features.

Instead, when deploying the model, we will use `dill` (a pickle extension) which allows us to pickle a function definition and use it elsewhere without having to strictly define imports when loading (assuming all modules are already installed).

The feature extraction function signature should be:
```python
input_features = get_features(y, sr, hop_size=None)
```

The pickle file should be saved alongside the model sharing the same name, just with the `.pkl` extension.

For an example: `models/quantized_default_int8.tflite` -> `models/quantized_default_int8.pkl`

For an example of how to define your own feature extractor, see `sonyc_ml/features.py`. Try to keep the feature extractor dependencies basic so we only load what dependencies we need for when we import it later, and try to only include modules that are used across `sonycnode` (e.g. librosa, numpy).

The current feature extraction function was exported using:
```bash
python sonyc_ml/features.py export \
    path/to/quantized_default_int8.pkl \
    --duration=1 --hop_size=0.1 \
    --target_sr=8000 \
    --n_fft=1024 \
    --n_mels=64 \
    --mel_hop_len=160 \
    --fmax=None
```

## Models

There are 3 models defined in `model.py`. The only difference between them is the feature extraction step.

#### Pickled Feature Extractor
Assuming you've followed the steps above and pickled your feature extraction function next to your model, you should use `model.AudioModel` which will call the pickled function and pass the result to the model.

If you haven't pickled the feature extractor yet and the model uses `features.get_features`, you can use `model.SpecModel`. Extra kw args in `__init__` are passed to `features.get_features(...)`.

#### Model takes raw PCM
I also left `model.PCMModel` which only performs windowing on the PCM data. This would be useful if you were using `kapre` in your models (or wavenet or something).
