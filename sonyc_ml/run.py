import os
import time

from model import AudioModel as Model

def watch(model_path, input_path='./data/audio', output_path='./data/l3',
          ext='.wav', hop_size=0.1, **kw):
    from osutil import filewatch
    l3 = Model(model_path, output_path=output_path, hop_size=hop_size, **kw)

    @filewatch(input_path)
    def run(event):
        if event.pathname.endswith(ext):
            l3.process_file(event.pathname)


def test(model_path, output_path=None, n=10, **kw):
    TEST_DIR = os.path.dirname(os.path.dirname(__file__))
    AUDIO_PATH = os.path.join(TEST_DIR, 'data', 'chirp_1s.wav')
    output_path = output_path or os.path.join(TEST_DIR, 'data', 'test_output')
    print('Input Audio File:', AUDIO_PATH)
    print('Output Dir:', output_path)

    model = Model(model_path, output_path=output_path, **kw)

    t0 = time.time()
    for i in range(n):
        ti = time.time()
        model.process_file(AUDIO_PATH)
        print('Inference run {}/{}: {:.3f}s'.format(i + 1, n, time.time() - ti))
    print('Total run time ({} iter): {:.3f}s'.format(n, time.time() - t0))

if __name__ == '__main__':
    import fire
    fire.Fire()
