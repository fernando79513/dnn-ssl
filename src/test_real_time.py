from src.utils.emd import pit_earth_mover_distance
import numpy as np
import matplotlib.pyplot as plt
import json

# import tensorflow as tf
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle

from src.cnn_blstm_ss import encode_angles, soft_encode_2_angles
from utils import phase_map

if __name__ == "__main__":


    predictions = model.predict(s_pmaps[:20])
    print("predictions shape:", predictions.shape)

    for i in range(20):
        plt.plot(s_labels[i])
        plt.plot(predictions[i])
        plt.show()
        # plt.savefig(f"img/pred_{1:0>4}")


import cv2 as cv
import numpy as np


size = (400,1440)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('sources.avi', fourcc, 15, (360, 1440))

for i in range(300):
    img = np.full((100*4,360*4,3), 255, np.uint8)
    cv.circle(img,(int(3*400), 60*4), 5, (150,150,40), -1)
    cv.circle(img,(int(3*400), 150*4), 5, (40,150,150), -1)

    out.write(img)

out.release()
