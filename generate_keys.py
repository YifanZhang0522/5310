import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ['Emily Johnson', 'Alex Smith', 'Laura Miller', 'Yifan Zhang']
usernames = ['EmilyJ_88', 'AlexS_45', 'LauraM_23', 'YifanZ_22']
passwords = ['abc123', 'def456', 'ghi789', 'jkl000']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / 'hashed_pw.pk1'

with file_path.open('wb') as file:
    pickle.dump(hashed_passwords, file)