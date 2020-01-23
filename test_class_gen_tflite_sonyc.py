#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import csv
import json
import glob
import numpy as np
import resampy
import tensorflow as tf
import soundfile as sf
import librosa
import time

# In[8]:


def get_output_path(filepath, suffix, output_dir=None):
    """
    Parameters
    ----------
    filepath : str
        Path to audio file to be processed
    suffix : str
        String to append to filename (including extension)
    output_dir : str or None
        Path to directory where file will be saved. If None, will use directory of given filepath.
    Returns
    -------
    output_path : str
        Path to output file
    """
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    if not output_dir:
        output_dir = os.path.dirname(filepath)

    if suffix[0] != '.':
        output_filename = "{}_{}".format(base_filename, suffix)
    else:
        output_filename = base_filename + suffix

    return os.path.join(output_dir, output_filename)


# In[10]:


def _center_audio(audio, frame_len):
    """Center audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)


def _pad_audio(audio, frame_len, hop_len):
    """Pad audio if necessary so that all samples are processed"""
    audio_len = audio.size
    if audio_len < frame_len:
        pad_length = frame_len - audio_len
    else:
        pad_length = int(np.ceil((audio_len - frame_len)/float(hop_len))) * hop_len                      - (audio_len - frame_len)

    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    return audio

def _amplitude_to_db(S, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)
    return log_spec


# In[24]:


def get_softmax_from_tflite(audio, sr, model_path=None, hop_size=0.1, center=True,                            n_fft=None, n_mels=None, mel_hop_len=None, fmax=None):
    """
    Computes and returns L3 embedding for given audio data
    """
    if model_path is None:
        raise ValueError('Tflite Model Path is missing')
    
    if audio.size == 0:
        raise ValueError('Got empty audio')

    # Resample if necessary
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=TARGET_SR, filter='kaiser_best')

    audio_len = audio.size
    frame_len = TARGET_SR
    hop_len = int(hop_size * TARGET_SR)

    if audio_len < frame_len:
        warnings.warn('Duration of provided audio is shorter than window size (1 second). Audio will be padded.',
                      L3Warning)

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    frames = librosa.util.utils.frame(audio, frame_length=frame_len, hop_length=hop_len).T
    X = []
    for frame in frames:
        S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_len,                                     window='hann', center=True, pad_mode='constant'))
        S = librosa.feature.melspectrogram(sr=sr, S=S, n_mels=n_mels, fmax=fmax,
                                           power=1.0, htk=True)
        S = _amplitude_to_db(np.array(S))
        X.append(S)

    # Get the output for each frame
    batch_size = min(len(X), 64)

    predictions = []
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:]
    output_shape = output_details[0]['shape'][1:]
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    #Reshape the tensor so that tflite can predict in batches
    interpreter.resize_tensor_input(input_index, ((batch_size, ) + tuple(input_shape)))
    interpreter.resize_tensor_input(output_index, ((batch_size, ) + tuple(input_shape)))
    interpreter.allocate_tensors()
    
    print("== Input details ==")
    print(interpreter.get_input_details()[0])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print(interpreter.get_output_details()[0])
   
    #Predict for one audio file i.e. for one batch
    for cnt in range(10):
        st = time.time()
        x = np.array(X[:batch_size])[:, :, :, np.newaxis].astype(np.float32)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        predictions.append(output)
        print('Inference run %i: %0.3f' % (cnt + 1, time.time() - st))
#     predictions per batch in the absence of reshape   
#     for idx, batch_x in enumerate(batch_size):
#         x = np.array(X[idx])[np.newaxis, :, :, np.newaxis].astype(np.float32)
#         interpreter.set_tensor(input_index, x)
#         interpreter.invoke()
#         output = interpreter.get_tensor(output_index)
#         predictions.append(output)
    
    return predictions


# In[14]:


def process_file(filepath, output_dir=None, model_path=None, hop_size=0.1,                 n_fft=None, n_mels=None, mel_hop_len=None, fmax=None):
    """
    Computes and saves L3 embedding for given audio file
    """
    if not os.path.exists(filepath):
        raise ValueError('File "{}" could not be found.'.format(filepath))

    try:
        audio, sr = sf.read(filepath)
    except Exception:
        raise ValueError('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))

    output_path = get_output_path(filepath, ".npz", output_dir=output_dir)

    output = get_softmax_from_tflite(audio, sr, model_path=model_path, hop_size=hop_size,                                     n_fft=n_fft, n_mels=n_mels, mel_hop_len=mel_hop_len, fmax=fmax)

    #coarse classes of sonyc = 8
    pred = np.array(output).reshape(-1, 8)
    #Get the position (or class) corresponding to the maximum output
    pred_max = np.argmax(pred, axis=1)
    
    np.savez(output_path, label=pred_max)
    assert os.path.exists(output_path)


# In[32]:


if __name__=='__main__':
    TEST_DIR = os.path.dirname(os.path.abspath('__file__')) #os.path.dirname(__file__)
    TEST_DIR ='mnt/pi_ml_container'
    TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data/8k')
    TFLITE_MODELS_DIR = os.path.join(TEST_DIR, 'tflite_models')
    OUTPUT_DIR = os.path.join(TEST_DIR, 'output/sonyc_labels')
    
    MODEL_PATH = os.path.join(TFLITE_MODELS_DIR, 'full_quantized_default_float32.tflite')
    SAMP_8K_PATH_1 = os.path.join(TEST_AUDIO_DIR, '08_003165.wav')
    SAMP_8K_PATH_2 = os.path.join(TEST_AUDIO_DIR, '34_000997.wav')

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    TARGET_SR = 8000
    n_mels = 64
    hop_size = 0.1 
    mel_hop_len = 160
    n_fft = 1024
    fmax = None
    
    process_file(SAMP_8K_PATH_1, output_dir=OUTPUT_DIR, model_path=MODEL_PATH, hop_size=hop_size, n_mels=n_mels, n_fft=n_fft, mel_hop_len=mel_hop_len, fmax=fmax)
    
    # Labels are genrated for each frame. You can read the labels 
    labels = np.load(os.path.join(OUTPUT_DIR, '08_003165.npz'))
    assert labels['label'].shape[0] == 64
    print(labels['label'])

