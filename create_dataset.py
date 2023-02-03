from utils.dataset_utils import create_dataset, load_reference_signs, load_dataset
from sign_recorder import SignRecorder
import pickle

create_dataset()

videos = load_dataset()

reference_signs = load_reference_signs(videos)

sign_recorder = SignRecorder(reference_signs)

with open('ref.pkl', 'wb') as outp:
    pickle.dump(sign_recorder, outp, pickle.HIGHEST_PROTOCOL)