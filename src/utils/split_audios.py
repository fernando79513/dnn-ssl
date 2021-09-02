import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal

FS = 16000

fer_clean_start = [
    10.4,  40.5,    71, 100.6, 130.6, 160.8, 190.9,   221,   251, 281.2,  
    311.2, 341.3, 371.3, 401.5, 431.4, 461.6, 491.7, 521.8, 551.6, 581.8,
      612, 642.1, 672.1, 702.3]

fer_clean_end = [
    27.1,  54.9,  89.5, 118.7, 148.9, 180.3, 210.3, 239.1, 259.6, 298.2,
    330.7, 360.5, 391.1, 419.2, 450.1, 479.3,   499, 539.4, 567.7, 600.4, 
    629.6, 659.2, 691.3, 721.8]
car_clean_start = [
    10.3, 40.5,  70.4,  107.3,   137, 167.5, 190.8, 221.6, 251, 281,
    311.1, 341.2, 371.5, 401.3, 431.3, 460.8, 490.8, 520.1, 551, 581,
    611.2, 641.2, 671.3, 701.4]
car_clean_end = [
       27,  57.8,  85.6, 120.1, 149.3, 178.3, 210.4, 240.5, 270.7, 298.2,
    330.6, 359.5, 386.5, 420.8, 451.2, 480.4, 510.6, 540.7, 569.6,   600,
    630.9, 655.8, 690.8, 716.3]
two_clean_start = [
     10.4,  40.5,  70.6, 100.7, 130.8, 160.7, 190.8, 221.1,   251, 281.1,
    311.3, 341.5, 371.5, 401.5, 431.5, 461.6, 491.7, 521.8, 551.9, 581.9,
      612,   642, 672.1, 702.1, 732.3, 762.9, 792.5, 822.4]
two_clean_end = [
       28,  59.5,  86.7, 117.9, 150.3, 177.9, 208.7, 238.8, 270.7, 299.6,
    330.8, 360.5, 390.7, 417.1, 451.1, 481.2, 508.8, 539.1, 571.3, 599.7,
    631.5, 657.4, 689.7, 718.9, 752, 781.4, 810.6, 840]

if __name__ == "__main__":
    array = "kinect"
    filename = f"real_audio/2_sources/processed/recording_{array}_2src_clean_1.wav"
    fs, wav = wavfile.read(filename)
    print(wav.shape)
    print(702.3*16000)
    i =1
    for start, end in zip(car_clean_start, car_clean_end):
        start_i = int(start*FS)
        end_i = int(end*FS)
        short = wav[start_i:end_i,:]
        wavfile.write(f'real_audio/kinect/2_src/clean_{i:0>4}.wav',
            FS, short)
        i += 1


    import sounddevice as sd
    print('playing sound using  pydub')
    sd.play(short[:,0], fs)
    # sd.play(mono, fs)
    sd.wait()