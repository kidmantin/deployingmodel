import numpy as np
import bentoml

# BENTO_MODEL_TAG = "audio_commands_model_bento:zqr576qy5kzktmt2"

# audio_commands_model_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()

# svc = bentoml.Service("audio_commands_model", runners=[audio_commands_model_runner])

# @svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
# def predict(input_data: np.)