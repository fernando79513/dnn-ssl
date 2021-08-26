import logging

def set_logs():
    logging.basicConfig(filename='dnn.log', 
        format='%(asctime)s %(message)s', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return

def log_simulation(wav_file, n_speakers, n_noises, params):
    mic_name = params['mic_array']['name']
    speaker_count = n_speakers
    noise_count = n_noises
    # mic_pairs = [[1,4], [2,5], [3,6], [4,7], [5,1], [6,2], [7,3]]

    logging.info('-----------------Start Processing-----------------')    
    logging.info(f'Processing: {wav_file}')
    logging.info(f'Nº Speakers: {speaker_count} Nº Noise: {noise_count}')
    # logging.info(f'Length: {length}\tMax_tau: {max_tau}\tInterpol: {interp}' )
    # logging.info(f'Mic_pairs: {mic_pairs}')
    logging.info(f'Microphone Array - {mic_name}')
    return

    