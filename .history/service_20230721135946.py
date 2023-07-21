import numpy as np
import tensorflow as tf
import bentoml

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def preprocess_file(waveform):
    spectrogram = get_spectrogram(waveform).numpy()
    return np.expand_dims(spectrogram, axis=0) # return in batched form

BENTO_MODEL_TAG = "audio-commands-model-bentoml:latest"

audio_commands_model_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()

svc = bentoml.Service("audio_commands", runners=[audio_commands_model_runner])

# @svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
# def classify(input_data: np.ndarray) -> np.ndarray:
#     print(type(input_data))
#     processed_input = training.preprocess_file(input_data) # <class 'tensorflow.python.framework.ops.EagerTensor'>
#     # return audio_commands_model_runner.predict.run(processed_input)
#     return audio_commands_model_runner.run(processed_input) # np.ndarray


@svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    processed_input = preprocess_file(input_data) # <class 'tensorflow.python.framework.ops.EagerTensor'>
    return audio_commands_model_runner.run(processed_input) # np.ndarray