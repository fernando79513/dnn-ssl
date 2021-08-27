import numpy as np
import matplotlib.pyplot as plt
import json

# import tensorflow as tf
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle

from src.cnn_blstm_ss import encode_angles

import wandb
from wandb.keras import WandbCallback



def test_model(model,x,y):
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred,axis=1)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y, y_pred))

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    from src.combine_data import shuffle_df

    config_file = 'config/param_matrix_voice.json'
    with open(config_file) as f:
        params = json.load(f)
    mic_pairs = params['gcc_phat']['mic_pairs']

    model = tf.keras.models.load_model('data/matrix_voice/models/blstm.h5')
    print('model loaded')
    checkpoint_filepath = 'data/matrix_voice/checkpoint/'
    model.load_weights(checkpoint_filepath)
    print('weights loaded')

    gamma = 10
    pmaps = np.load('data/matrix_voice/test/pmap_1_src_clean.npy')
    stft_data = np.load('data/matrix_voice/test/stft_data_1_src_clean.npy')

   
    pmaps = np.moveaxis(pmaps, [0,1,2,3], [0,-2,-1,-3])
    angles = stft_data[:, -2:]
    test_labels = encode_angles(angles, gamma)
    print(test_labels[:5])

    s_pmaps, s_labels = shuffle(pmaps, test_labels)

    # print("Evaluate on test data")
    # results = model.evaluate(test_inputs, test_labels, batch_size=128)
    # print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(s_pmaps[:20])
    print("predictions shape:", predictions.shape)

    for i in range(20):
        plt.plot(s_labels[i])
        plt.plot(predictions[i])
        plt.show()
        # plt.savefig(f"img/pred_{1:0>4}")
