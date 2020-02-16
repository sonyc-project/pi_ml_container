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
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
# New modules: oyaml and pandas
import oyaml as yaml

def _center_audio(audio, frame_len):
    """Center audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)


def _pad_audio(audio, frame_len, hop_len):
    """Pad audio if necessary so that all samples are processed"""
    audio_len = audio.size
    if audio_len < frame_len:
        pad_length = frame_len - audio_len
    else:
        pad_length = int(np.ceil((audio_len - frame_len)/float(hop_len))) * hop_len \
                     - (audio_len - frame_len)

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
    
def get_output_from_tflite(audio, sr, interpreter, input_index, output_index, hop_size=0.1, center=True,\
                            n_fft=None, n_mels=None, mel_hop_len=None, fmax=None):
    """
    Computes and returns output of the classifier
    """
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
        S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_len,\
                                     window='hann', center=True, pad_mode='constant'))
        S = librosa.feature.melspectrogram(sr=sr, S=S, n_mels=n_mels, fmax=fmax,
                                           power=1.0, htk=True)
        S = _amplitude_to_db(np.array(S))
        X.append(S)
        
    predictions = []
    
    #predictions per frame   
    for idx in range(len(X)):
        x = np.array(X[idx])[np.newaxis, :, :, np.newaxis].astype(np.float32)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        predictions.append(output)
    
    return predictions    

def write_to_output(output_path, test_file_list, y_pred, taxonomy):
    
    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                           for fine_id, fine_label in fine_dict.items()]
                          for coarse_id, fine_dict in taxonomy['fine'].items()]
        
    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                               for fine_label in fine_list]
        
    coarse_target_labels = ["_".join([str(k), v])
                            for k, v in taxonomy['coarse'].items()]
        
    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename]

            # Add placeholder values for fine level
            row += [0.0 for _ in range(len(full_fine_target_labels))]
            # Add coarse level labels
            row += list(y)

            csvwriter.writerow(row)
            
def process_files(file_list, taxonomy, output_path, model_path=None, hop_size=0.1,\
                 n_fft=None, n_mels=None, mel_hop_len=None, fmax=None):
    """
    Computes and saves L3 embedding for audio files
    """
    interpreter = tf.lite.Interpreter(model_path=model_path) 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:]
    output_shape = output_details[0]['shape'][1:]
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    interpreter.allocate_tensors()
    
    y_pred_mean = []
    for file in file_list:
        try:
            audio, sr = sf.read(file)
        except Exception:
            raise ValueError('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))

        output = get_output_from_tflite(audio, sr, interpreter, input_index, output_index, hop_size=hop_size,\
                                        n_fft=n_fft, n_mels=n_mels, mel_hop_len=mel_hop_len, fmax=fmax)

        #coarse classes of sonyc = 8
        pred_frame = np.array(output).reshape(-1, 8)
        y_pred_mean.append(pred_frame.mean(axis=0).tolist())
    
    write_to_output(output_path, file_list, y_pred_mean, taxonomy)
    assert os.path.exists(output_path)
    
def evaluate_all(prediction_path, annotation_path, yaml_path, mode='coarse'):
    
    metrics = {
        'coarse': {}
    }

    df_dict = evaluate(prediction_path,
                       annotation_path,
                       yaml_path,
                       mode)

    micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
    macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)    

     # Get index of first threshold that is at least 0.5
    thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]

    metrics[mode]["micro_auprc"] = micro_auprc
    metrics[mode]["micro_f1"] = eval_df["F"][thresh_0pt5_idx]
    metrics[mode]["macro_auprc"] = macro_auprc

    print("{} level evaluation:".format(mode.capitalize()))
    print("======================")
    print(" * Micro AUPRC:           {}".format(metrics[mode]["micro_auprc"]))
    print(" * Micro F1-score (@0.5): {}".format(metrics[mode]["micro_f1"]))
    print(" * Macro AUPRC:           {}".format(metrics[mode]["macro_auprc"]))
    print(" * Coarse Tag AUPRC:")

    metrics[mode]["class_auprc"] = {}
    for coarse_id, auprc in class_auprc.items():
        coarse_name = taxonomy['coarse'][int(coarse_id)]
        metrics[mode]["class_auprc"][coarse_name] = auprc
        print("      - {}: {}".format(coarse_name, auprc))       
        
if __name__=='__main__':
    
    TARGET_SR = 8000
    n_mels = 64
    hop_size = 0.1 
    mel_hop_len = 160
    n_fft = 1024
    fmax = None
    sample_test = False
    
    TEST_DIR = os.path.dirname(os.path.realpath('__file__'))
    TFLITE_MODELS_DIR = os.path.join(TEST_DIR, 'tflite_models')
    MODEL_PATH = os.path.join(TFLITE_MODELS_DIR, 'cmsis_mels_full_quantized_default_float32.tflite')
    OUTPUT_DIR = os.path.join(TEST_DIR, 'output/sonyc_ust/rpi_test')
        
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if sample_test:
        TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data/8k')
        SAMP_8K_PATH_1 = os.path.join(TEST_AUDIO_DIR, '08_003165.wav')
        SAMP_8K_PATH_2 = os.path.join(TEST_AUDIO_DIR, '34_000997.wav')
        
        process_file(SAMP_8K_PATH_1, output_dir=OUTPUT_DIR, model_path=MODEL_PATH,\
                     hop_size=hop_size, n_mels=n_mels, n_fft=n_fft, mel_hop_len=mel_hop_len,\
                     fmax=fmax)
    else:
        #Change the path to where you have the downloaded data
        AUDIO_DIR = '/beegfs/dr2915/sonyc_ust'
        TEST_AUDIO_DIR = os.path.join(AUDIO_DIR, 'test')
        annotation_path = os.path.join(AUDIO_DIR, 'annotations.csv')
        yaml_path = os.path.join(AUDIO_DIR, 'dcase-ust-taxonomy.yaml')
        test_files = glob.glob(TEST_AUDIO_DIR + '/*.wav')
        prediction_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
        output_path = os.path.join(OUTPUT_DIR, 'output_mean.csv')
    
        with open(yaml_path) as f:
            taxonomy = yaml.load(f, Loader=yaml.FullLoader)
        
        process_files(test_files, taxonomy, output_path, model_path=MODEL_PATH,\
                     hop_size=hop_size, n_mels=n_mels, n_fft=n_fft, mel_hop_len=mel_hop_len,\
                     fmax=fmax)
            
        evaluate_all(output_path, annotation_path, yaml_path)        


