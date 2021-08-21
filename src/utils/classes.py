from xml.dom import minidom
import numpy as np

class Speaker:
    def __init__(self, speaker):
        self.name = speaker
        # ? add cmu_arctic as variable
        self.dir = f'wav/cmu_arctic/{speaker}'
        self.id = 0
        self.audio = ""
        self.audio_ids = []

        self.polar_pos = []
        self.position = []

class Noise:
    def __init__(self, id):
        self.id = id
        self.dir = f'wav/noise/dns_{id:0>4}.wav'
        self.audio = ""

        self.polar_pos = []
        self.position = []

class MicArray:
    def __init__(self, name):
        self.name = name
        self.mic_count = 0
        self.positions = np.zeros((2,8))
        self.x_pos = np.zeros(8)
        self.y_pos = np.zeros(8)

        self.get_mics(name)

    def get_mics(self, name='matrix_voice'):
        # Read the microphone array
        # parsee an xml file by name
        mydoc = minidom.parse(f'config/{name}.xml')
        items = mydoc.getElementsByTagName('position')
        self.mic_count = len(items)
        self.x_pos = np.zeros(self.mic_count)
        self.y_pos = np.zeros(self.mic_count)

        for i, elem in enumerate(items):  
            self.x_pos[i] = float(elem.attributes['x'].value)
            self.y_pos[i] = float(elem.attributes['y'].value)

        self.positions = np.array([self.x_pos, self.y_pos])

        return
