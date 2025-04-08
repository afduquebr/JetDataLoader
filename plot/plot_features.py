import tables as tb
import numpy as np
import matplotlib.pyplot as plt

path = "/AtlasDisk/user/duquebran/JetTagging/5-classes/data_train/train_5M.h5"

# labels = [
#             'label_QCD', 
#             'label_WZ', 
#             'label_top', 
#             'label_higgs'
#         ]

labels = ['label_QCD', 'label_W', 'label_Z', 'label_top', 'label_higgs']

# Open the source HDF5 file
file = tb.open_file(path, mode="r")

# Get the indices where each label is 1
idx = [np.where(file.root[label][:] == 1)[0] for label in labels]

# Plot the distribution for each class
feature = 'jet_pt'

plt.figure(figsize=(15, 8))
for label, id in zip(labels, idx):
    plt.hist(file.root[feature][:][id], bins=100, weights=file.root['weight'][:][id], alpha=0.5, label=label, histtype='bar')
    # plt.hist(file.root[feature][:][id], bins=100, alpha=0.5, label=label, histtype='bar')

plt.xlabel('Jet pT')
plt.ylabel('counts')
# plt.yscale('log')
plt.title('Jet pT Distribution of 4 Classes')
plt.legend()
plt.grid(True)

# Save the figure as a PNG file
plt.savefig(f'{feature}_distribution.png', dpi=300)  # dpi=300 for high resolution