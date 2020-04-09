from functools import partial
import numpy as np
import librosa
import util as u

framepadlen = lambda xlen, flen, hlen: int(
    np.ceil(1. * (xlen - flen) / hlen) * hlen + flen)

def get_logmelspec(audio, sr, hop_size=0.1, duration=1, target_sr=8000, **kw):
    '''Computes and returns L3 input features for given audio data.'''
    assert audio.size, 'Got empty audio'
    assert duration == 1, 'For now, duration is fixed at 1 second'
    if sr and target_sr and sr != target_sr:
        audio, sr = librosa.core.resample(audio, sr, target_sr), target_sr

    frame_len = int(duration * sr)
    hop_len = int(hop_size * sr)

    # Pad if necessary to ensure that we process all samples
    # Split audio into frames, copied from librosa.util.frame
    # get spectrogram frames
    padlen = framepadlen(audio.size, frame_len, hop_len)
    audio = librosa.util.pad_center(audio, padlen)
    frames = librosa.util.frame(audio, frame_len, hop_len).T
    return u.npgenarray((_get_spec(frame, sr, **kw) for frame in frames), len(frames))

def _get_spec(frame, sr, n_fft=1024, n_mels=64,
              mel_hop_len=160, fmax=None):
    # magnitude spectrum
    S = np.abs(librosa.core.stft(
        frame, n_fft=n_fft, hop_length=mel_hop_len,
        window='hann', center=True, pad_mode='constant'))

    # log mel spectrogram
    return librosa.power_to_db(
        librosa.feature.melspectrogram(
            S=S, sr=sr, n_mels=n_mels, fmax=fmax, htk=True), amin=1e-10)


def export(output_path, **kw):
    import dill
    dill.settings['recurse'] = True
    with open(output_path, 'wb') as f:
        dill.dump(partial(get_logmelspec, **kw), f)

if __name__ == '__main__':
    import fire
    fire.Fire()
