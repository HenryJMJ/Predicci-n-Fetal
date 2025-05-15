import matplotlib.pyplot as plt
import tensorflow as tf

# Forzar carga de backend y caché
plt.figure()
plt.close()

# Forzar inicialización de TF para CPU
tf.constant(1)
