# import pandas as pd
# import numpy as np
# import time
# from sklearn.model_selection import train_test_split
import tensorflow as tf

if __name__ == '__main__':

    version = tf.__version__
    gpu_ok = tf.test.is_gpu_available()
    print("tf version:", version, "\nuse GPU", gpu_ok)
