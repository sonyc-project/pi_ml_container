import itertools
import numpy as np

'''

Model

'''

def load_tflite_model_function(model_path):
    import tflite_runtime.interpreter as tflite
    compute = prepare_model_function(tflite.Interpreter(model_path))
    compute.model_path = model_path
    return compute


def prepare_model_function(model, verbose=False):
    # allocate and get shapes
    model.allocate_tensors()
    in0_dets = model.get_input_details()[0]
    out0_dets = model.get_output_details()[0]
    output_shape = out0_dets['shape'][1:]
    input_index, output_index = in0_dets['index'], out0_dets['index']

    if verbose:
        print('-- Input details --')
        print(in0_dets, '\n')
        print('-- Output details --')
        print(out0_dets, '\n')

    # Get the L3 embedding
    def compute(x):
        x = x[None, :, :, None].astype(np.float32)
        model.set_tensor(input_index, x)
        model.invoke()
        Z = model.get_tensor(output_index)
        return np.reshape(Z, (Z.shape[0], Z.shape[-1]))

    compute.model = model
    compute.output_shape = output_shape
    return compute


'''

Utils

'''

def precheck_iter(it, n=1):
    '''Check the value first n items of an iterator without unloading them from
    the iterator queue.'''
    it = iter(it)
    items = [_ for _, i in zip(it, range(n))]
    return items, itertools.chain(items, it)

def npgenarray(it, shape, **kw):
    '''Create a np.ndarray from a generator. Must specify at least the length
    of the generator or the entire shape of the final array.'''
    if isinstance(shape, int):
        (x0,), it = precheck_iter(it)
        shape = (shape,) + x0.shape
    X = np.zeros(shape, **kw)
    for i, x in enumerate(it):
        X[i] = x
    return X
