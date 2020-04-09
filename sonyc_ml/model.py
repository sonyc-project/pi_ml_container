import os
import numpy as np
import librosa

from util import load_tflite_model_function, npgenarray
from osutil import get_output_path


class PCMModel:
    compute = model_path = None

    def __init__(self, model_path, output_path=None, suffix='', target_sr=None,
                 duration=1, hop_size=0.1):
        self.load_model(model_path)
        self.output_path = output_path
        self.suffix = suffix
        self.target_sr = target_sr
        self.duration = duration
        self.hop_size = hop_size

    def load_model(self, model_path=None):
        '''Load a tflite model from file'''
        self.model_path = model_path or self.model_path
        self.compute = load_tflite_model_function(self.model_path)
        return self.compute

    def get_features(self, y, sr=None, hop_size=None):
        '''Return the input features for the model given the audio pcm and sample rate.'''
        if sr and self.target_sr:
            y, sr = librosa.resample(y, sr, self.target_sr), self.target_sr

        return librosa.util.frame(
            y, self.duration * (self.target_sr or sr),
            hop_size or self.hop_size)

    def get_embedding(self, y, sr=None, hop_size=None):
        '''Get the embedding from the audio pcm and sample rate.'''
        X = self.get_features(y, sr, hop_size=hop_size)
        # compute the embeddings for each frame
        return npgenarray((self.compute(x) for x in X), len(X))

    def save_embedding(self, embedding, input_path=None, output_path=None):
        '''Save the embedding to .npz given the input path and the embedding.
        The output path will be calculated from the input_path. If output_path
        is specified, it will be used as is.
        '''
        assert output_path or input_path
        output_path = output_path or get_output_path(
            input_path, (self.suffix or '') + '.npz', self.output_path)

        np.savez(output_path, embedding=embedding)
        assert os.path.exists(output_path)
        return output_path

    def process_file(self, audio_path, output_path=None, hop_size=None):
        '''Load audio from file, compute embeddings, and save to .npz.'''
        y, sr = librosa.load(audio_path, sr=self.target_sr)
        embedding = self.get_embedding(y, sr, hop_size)
        return self.save_embedding(embedding, audio_path, output_path)


class SpecModel(PCMModel):
    def __init__(self, model_path, output_path=None, suffix='',
                 duration=1, hop_size=0.1, target_sr=8000,
                 n_fft=1024, n_mels=64, mel_hop_len=160,
                 fmax=None):
        super().__init__(
            model_path, output_path, suffix,
            hop_size=hop_size, target_sr=target_sr,
            duration=duration)

        # define feature extractor
        from features import get_logmelspec
        self.get_features = lambda audio, sr=None, hop_size=None: (
            get_logmelspec(
                audio, sr, n_fft=n_fft, n_mels=n_mels,
                mel_hop_len=mel_hop_len, fmax=fmax,
                hop_size=hop_size or self.hop_size, target_sr=target_sr,
                duration=duration))


class AudioModel(PCMModel):
    _get_features = None

    def __init__(self, model_path, output_path=None, suffix='',
                 duration=1, hop_size=0.1, target_sr=48000):
        super().__init__(
            model_path, output_path, suffix,
            hop_size=hop_size, target_sr=target_sr,
            duration=duration)
        self.load_feature_extractor(os.path.splitext(model_path)[0] + '.pkl')

    def get_features(self, y, sr, hop_size=None):
        return self._get_features(y, sr, hop_size=hop_size or self.hop_size)

    def load_feature_extractor(self, pkl_path):
        '''Load the feature extraction function, get_features, from pickle file.'''
        import dill
        with open(pkl_path, 'rb') as f:
            self._get_features = dill.load(f)
