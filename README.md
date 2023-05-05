# DNN SSL

This module has a neural network implementation for sound source separation


The neural network architecture used for sound source localization is inspired by the D-ASR network. This network consists of two sections, one with convolutional neural networks (CNNs) and the other with BLSTM networks. The first section takes the phase map of an audio segment as input. This phase map corresponds to the imaginary part of the STFT of that segment. Each microphone produces a phase map, and the convolutional network reduces them to a single tensor.