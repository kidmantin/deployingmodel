import numpy as np
import bentoml
import training

BENTO_MODEL_TAG = "audio_commands_model_bento:zqr576qy5kzktmt2"

audio_commands_model_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()

svc = bentoml.Service("audio_commands_model", runners=[audio_commands_model_runner])

# @svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
# def classify(input_data: np.ndarray) -> np.ndarray:
#     print(type(input_data))
#     processed_input = training.preprocess_file(input_data) # <class 'tensorflow.python.framework.ops.EagerTensor'>
#     # return audio_commands_model_runner.predict.run(processed_input)
#     return audio_commands_model_runner.run(processed_input) # np.ndarray

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    input_data = np.array(input_data)
    # print(type(input_data))
    processed_input = training.preprocess_file(input_data) # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # return audio_commands_model_runner.predict.run(processed_input)
    return audio_commands_model_runner.run(processed_input) # np.ndarray