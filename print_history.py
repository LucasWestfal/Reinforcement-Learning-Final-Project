import numpy as np

# Specify the path to your .npy file
file_path = 'models/MATD3/training_scores_history.npy' 

# Load the .npy file
data = np.load(file_path)

# Now 'data' contains the NumPy array stored in the .npy file
print(data)