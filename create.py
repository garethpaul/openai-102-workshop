import numpy as np
import pickle
#
sample_saved_embeddings = [
    (1, np.array([0.1, 0.2]), {'text': 'sample text'})]

# save the sample_saved_embeddings to a pickle file
with open('test_embeddings.pkl', 'wb') as file:
    pickle.dump(sample_saved_embeddings, file)
