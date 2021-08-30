import os
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import categorical_crossentropy

def earth_mover_distance(
        **kwargs
) -> Callable:
    """
    Wrapper for earth_mover distance for unified interface with self-guided earth mover distance loss.
    """
    def _earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:
        return tf.reduce_sum(tf.square(tf.cumsum(y_true, axis=-1) - tf.cumsum(y_pred, axis=-1)), axis=-1)

    return _earth_mover_distance

def roll_earth_mover_distance(
        **kwargs
) -> Callable:
    """
    Wrapper for earth_mover distance for unified interface with self-guided earth mover distance loss.
    """
    def _roll_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:

        def emd(true, pred):
            emd_t = tf.reduce_sum(tf.square(tf.cumsum(
                true, axis=-1) - tf.cumsum(pred, axis=-1)), axis=-1)
            return emd_t
        def center_roll(t_p):
            t, p =  tf.split(t_p, num_or_size_splits=2, axis=-1)
            shift = 179 - tf.argmax(t)
            t = tf.roll(t,shift,axis=-1)
            p = tf.roll(p,shift,axis=-1)
            loss = emd(t,p)
            return loss

        true_1, true_2 =  tf.split(y_true, num_or_size_splits=2, axis=1)
        pred_1, pred_2 =  tf.split(y_pred, num_or_size_splits=2, axis=1)
        emd_1_1 = tf.map_fn(fn=center_roll, 
            elems=tf.concat((true_1,pred_1), axis=-1))
        emd_2_2 = tf.map_fn(fn=center_roll, 
            elems=tf.concat((true_2,pred_2), axis=-1))

        roll_emd = emd_1_1 + emd_2_2

        return roll_emd

    return _roll_earth_mover_distance

def pit_earth_mover_distance(
        **kwargs
) -> Callable:
    """
    Wrapper for earth_mover distance for unified interface with self-guided earth mover distance loss.
    """
    def _pit_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:

        def emd(true, pred):
            emd_t = tf.reduce_sum(tf.square(tf.cumsum(
                true, axis=-1) - tf.cumsum(pred, axis=-1)), axis=-1)
            return emd_t
        def center_roll(t_p):
            t, p =  tf.split(t_p, num_or_size_splits=2, axis=-1)
            shift = 179 - tf.argmax(t)
            t = tf.roll(t,shift,axis=-1)
            p = tf.roll(p,shift,axis=-1)
            loss = emd(t,p)
            return loss

        true_1, true_2 =  tf.split(y_true, num_or_size_splits=2, axis=1)
        pred_1, pred_2 =  tf.split(y_pred, num_or_size_splits=2, axis=1)
        emd_1_1 = tf.map_fn(fn=center_roll, 
            elems=tf.concat((true_1,pred_1), axis=-1))
        emd_1_2 = tf.map_fn(fn=center_roll, 
            elems=tf.concat((true_1,pred_2), axis=-1))
        emd_2_1 = tf.map_fn(fn=center_roll, 
            elems=tf.concat((true_2,pred_1), axis=-1))
        emd_2_2 = tf.map_fn(fn=center_roll, 
            elems=tf.concat((true_2,pred_2), axis=-1))

        # cce_1 = tf.math.add(emd(true_1, pred_1), emd(true_2, pred_2))
        # cce_2 = tf.math.add(emd(true_1, pred_2), emd(true_2, pred_1))
        pit_emd = tf.math.minimum(emd_1_1 + emd_2_2, emd_1_2 + emd_2_1)

        return pit_emd

    return _pit_earth_mover_distance

def pit_cce(
        **kwargs
) -> Callable:
    """
    Wrapper for earth_mover distance for unified interface with self-guided earth mover distance loss.
    """
    def _pit_cce(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:

        true_1, true_2 =  tf.split(y_true, num_or_size_splits=2, axis=1)
        pred_1, pred_2 =  tf.split(y_pred, num_or_size_splits=2, axis=1)
        cce_1 = tf.math.add(categorical_crossentropy(true_1, pred_1),
            categorical_crossentropy(true_2, pred_2))
        cce_2 = tf.math.add(categorical_crossentropy(true_1, pred_2),
            categorical_crossentropy(true_2, pred_1))
        pit_cce = tf.math.minimum(cce_1, cce_2)

        return pit_cce

    return _pit_cce


def approximate_earth_mover_distance(
        entropic_regularizer: float,
        distance_matrix: np.array,
        matrix_scaling_operations: int = 100,
        **kwargs
) -> Callable:
    """
    Wrapper for approximate earth mover distance.
    """

    def _approximate_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:
        k = tf.exp(-entropic_regularizer * distance_matrix)
        km = k * distance_matrix
        u = tf.ones(y_true.shape) / y_true.shape[1]
        for _ in range(matrix_scaling_operations):
            u = y_pred / ((y_true / (u @ k)) @ k)
        v = y_true / (u @ k)
        return tf.reduce_sum(u * (v @ km)) / y_true.shape[0]

    return _approximate_earth_mover_distance


class EmdWeightHeadStart(Callback):
    """Class for implementing delayed inclusion of the distance-based regularization term for the self-guided emd."""

    def __init__(self):
        super(EmdWeightHeadStart, self).__init__()
        self.emd_weight = 0
        self.epoch = 0
        self.cross_entropy_loss_history = []
        self.self_guided_emd_loss_history = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        if epoch == 4:
            cross_entropy_loss = tf.reduce_mean(
                tf.convert_to_tensor(self.cross_entropy_loss_history, dtype=tf.float32)
            )
            self_guided_emd_loss = tf.reduce_mean(
                tf.convert_to_tensor(self.self_guided_emd_loss_history, dtype=tf.float32)
            )
            self.emd_weight = (cross_entropy_loss / self_guided_emd_loss) / 3.5


class GroundDistanceManager(Callback):
    """Class for managing the generation and update of the ground distance matrix."""

    def __init__(
            self,
            file_path: Path
    ):
        super(GroundDistanceManager, self).__init__()
        self.ground_distance_matrix = None
        self.epoch_class_features = []
        self.epoch_labels = []
        self.class_length = 8

        file_path.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path

    def set_labels(self, labels):
        labels_tensor = tf.concat(labels, axis=0)
        self.epoch_labels = labels_tensor

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_class_features.append(self.model.second_to_last_layer)

    def on_epoch_end(self, epoch, logs=None):
        self._update_ground_distance_matrix()
        self._save_ground_distance_matrix(epoch=epoch)

    def _update_ground_distance_matrix(self):
        self.epoch_class_features = tf.concat(self.epoch_class_features, axis=0)
        estimated_distances = self._estimate_distances()
        self.ground_distance_matrix = self._calculate_ground_distances(
            estimated_distances=estimated_distances
        )
        self.epoch_class_features = []

    def _estimate_distances(self) -> K.placeholder:
        normalized_features = self.epoch_class_features \
                              / tf.reduce_sum(self.epoch_class_features, axis=1, keepdims=True)
        class_labels = K.argmax(self.epoch_labels, axis=-1)
        centroids = []
        for i in range(self.class_length):
            centroids.append(K.mean(normalized_features[class_labels == i], axis=0))
        centroids = tf.stack(centroids)
        tf.print(f'centroids: {centroids.numpy()}')
        estimated_distances = []
        for i in range(self.class_length):
            estimated_distances.append(
                tf.norm(
                    tensor=centroids - centroids[i],
                    ord='euclidean',
                    axis=-1
                )
            )
        return tf.stack(estimated_distances)

    def _calculate_ground_distances(
            self,
            estimated_distances: K.placeholder
    ) -> K.placeholder:
        tf.print(f'estimated_distances: {estimated_distances.numpy()}')
        sorted_indices = tf.argsort(estimated_distances)
        elements_smaller = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                elements_smaller[i, sorted_indices[i, j]] = j
        elements_smaller = tf.convert_to_tensor(elements_smaller, dtype=tf.float32)
        tf.print(f'elements_smaller: {elements_smaller.numpy()}')
        normalized_distances = (1 / (self.class_length - 1)) * elements_smaller
        return (normalized_distances + K.transpose(normalized_distances)) / 2

    def _save_ground_distance_matrix(self, epoch: int) -> None:
        np.save(
            file=str(self.file_path) + f'/{epoch}',
            arr=self.ground_distance_matrix
        )

    def load_ground_distance_matrix(self, epoch) -> np.array:
        return np.load(str(self.file_path) + f'/{epoch}.npy')


def self_guided_earth_mover_distance(
        model,
        ground_distance_sensitivity: float,
        ground_distance_bias: float
) -> Callable:
    """Wrapper for the self-guided earth mover distance loss function."""

    def _self_guided_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder,
    ) -> K.placeholder:
        cross_entropy_loss = categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred
        )
        if model.emd_weight_head_start.emd_weight == 0:
            if model.emd_weight_head_start.epoch == 3:
                self_guided_emd_loss = _calculate_self_guided_loss(
                    y_true=y_true,
                    y_pred=y_pred,
                    ground_distance_sensitivity=ground_distance_sensitivity,
                    ground_distance_bias=ground_distance_bias,
                    ground_distance_manager=model.ground_distance_manager
                )
                model.emd_weight_head_start.cross_entropy_loss_history.append(cross_entropy_loss)
                model.emd_weight_head_start.self_guided_emd_loss_history.append(self_guided_emd_loss)
            return cross_entropy_loss
        else:
            self_guided_emd_loss = _calculate_self_guided_loss(
                y_true=y_true,
                y_pred=y_pred,
                ground_distance_sensitivity=ground_distance_sensitivity,
                ground_distance_bias=ground_distance_bias,
                ground_distance_manager=model.ground_distance_manager
            )
            return cross_entropy_loss + model.emd_weight_head_start.emd_weight * self_guided_emd_loss

    return _self_guided_earth_mover_distance


def _calculate_self_guided_loss(
        y_true: K.placeholder,
        y_pred: K.placeholder,
        ground_distance_sensitivity: float,
        ground_distance_bias: float,
        ground_distance_manager: GroundDistanceManager
):
    batch_size = 32
    cost_vectors = []
    for i in range(batch_size):
        cost_vectors.append(
            ground_distance_manager.ground_distance_matrix[:, K.argmax(y_true[i])]
            ** ground_distance_sensitivity
            + ground_distance_bias
        )
    cost_vectors = tf.stack(cost_vectors)
    return K.sum(K.square(y_pred) * cost_vectors, axis=1)