import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forzar CPU

import tensorflow as tf
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model_rnn():
    return tf.keras.models.load_model("modelo_rnn.keras")

model = load_model_rnn()
