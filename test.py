from utils.dataset_utils import load_reference_signs
import pandas as pd
import pickle

ref = load_reference_signs(['a lot-3'])

print(type(ref.at[0,'sign_model']))
with open('ref.pkl', 'wb') as outp:
    company1 = ref.at[0,'sign_model']
    pickle.dump(company1, outp, pickle.HIGHEST_PROTOCOL)

del company1
with open('ref.pkl', 'rb') as inp:
    company1 = pickle.load(inp)
    print(company1.lh_embedding) 