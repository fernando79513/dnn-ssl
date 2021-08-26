import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import ast

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class LabelGenerator:
    def __init__(self, sim_df, params):
        self.sim_df = sim_df
        self.speaker_count = params['speakers']['count']
        self.noise_count = params['noises']['count']
        self.gcc_lenght = params['gcc_phat']['length']
        self.sig = 8
        
        self.wav_path = 'wav/cmu_arctic/'

    def encode_output(self, th=[]):
        angles = np.int_(np.linspace(0, 359, 360))
        if len(th) == 0:
            x = np.zeros((360,))
        else:
            x = np.empty((len(th),360))
            dist = gaussian(angles,180, self.sig)
            for i, mu in enumerate(th):
                x[i,:] = np.roll(dist, mu-180)
            x = np.max(x, axis=0)
        return x

    def prepare_output(self, index, info, angles, n_frames):
        # remove sources when there is no audio
        if self.speaker_count > 1:
            audio_frames = []
            for i in range(self.speaker_count):
                audio_length = 0
                name = info[f'speaker name {i}']
                audios = ast.literal_eval(info[f'speaker audios {i}'])
                for audio in audios:
                    wav_file = f'{self.wav_path}{name}/arctic_{audio}.wav'
                    wav = wavfile.read(wav_file)[1]
                    audio_length += len(wav)
                audio_frames.append(audio_length//self.gcc_lenght)
        else:
            audio_frames = [n_frames]
        
        labels = ['id', 'frame', 'number of speakers', 'number of noises']
        for j, angle in enumerate(angles):
            labels += [f'speaker angle {j}']
        for j in range(360):
            labels += [f'{j}']
        df_out = pd.DataFrame(columns=labels)

        # TODO this is dirty, try to think a better way
        for i in range(n_frames):
            data = [f'{index:0>4}', i, self.speaker_count, self.noise_count]
            current_angles = []
            for j, angle in enumerate(angles):
                if i < audio_frames[j]:
                    data += [angle]
                    current_angles.append(angle)
                else: 
                    data += [-1]

            angle_prob = self.encode_output(current_angles)
            for j in range(360):
                data += [angle_prob[j]]
            df_out.loc[i] = data
        return df_out

    def prepare_labels(self, index, n_frames):
        '''Gather the information from the simulation csv and prepare the
        labels for training. Take into account the duration of the speech
        to create the label of the speech segment'''
        info = self.sim_df.iloc[index-1]
        angles = []
        audios = []
        for i in range(self.speaker_count): 
            speaker_pos = np.array(info[f'speaker position {i}'][1:-1].split())
            speaker_pos = speaker_pos.astype(float).astype(int)
            angles.append(speaker_pos[0])
    
        # TODO See if it makes sense to add noise angles
        df_out = self.prepare_output(index, info, angles, n_frames)
        return df_out     



