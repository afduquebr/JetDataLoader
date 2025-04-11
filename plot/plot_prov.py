import tables 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Number of classes
num_classes = 4

# Path
path = f"/sps/atlas/a/aduque/JetTagging/{num_classes}-classes/data_train"
# path = f"/sps/atlas/a/aduque/JetTagging/{num_classes}-classes/data_train_new_reweight"

# Files
input_files = [
    f"{path}/train_files/train_QCD.h5",
    # f"{path}/train_files/train_WZ.h5",
    # f"{path}/train_files/train_top.h5",
    f"{path}/train_files/train_higgs.h5"
]

output_path = f"/pbs/home/a/aduque/private/JetDataLoader/figs/{num_classes}-classes/input_data/"

os.makedirs(output_path, exist_ok=True)

# labels = ['label_QCD', 'label_WZ', 'label_top', 'label_higgs']
labels = ['label_QCD', 'label_higgs']

files = [tables.open_file(input_file, mode="r") for input_file in input_files]

# idx = [np.where(file.root[label][:] == 1)[0] for file, label in zip(files, labels)]

colors = sns.color_palette("muted", len(labels))

plt.figure(figsize=(12, 7))

for label, color, file in zip(labels, colors, files):

    feature = 'jet_pt'
    data_sample = file.root[feature][:]
    is_multidim = len(data_sample.shape) > 1

    use_weights = (feature == 'jet_pt')

    data = data_sample

    # Flatten if multidimensional (e.g., (N_events, 80, 1) → (N_events * 80,))
    if is_multidim:
        data = data.reshape(-1)

    weights = None
    if use_weights:
        weights = file.root['weight'][:]
        if is_multidim:
            # Repeat weights per constituent
            weights = np.repeat(weights, data_sample.shape[1])

    # Exclude data equal to -999
    mask = data != -999
    data = data[mask]
    if weights is not None:
        weights = weights[mask]

    plt.hist(data,
                bins=400,
                alpha=0.2,
                label=label.replace('label_', ''),
                weights=weights,
                color=color,
                histtype='stepfilled',
                edgecolor='black',
                linewidth=1.2)

feature_name = feature.replace('_', ' ').title()

plt.xlabel(feature_name, fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.yscale('log')
# plt.xscale('log')
plt.title(f'{feature_name} Distribution (LOG binning Reweighting)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(f'{output_path}/{feature}_distribution_new.pdf', dpi=300, transparent=True)
plt.close()

jet_pt = file.root['jet_pt'][:]
weights = file.root['weight'][:]


plt.figure(figsize=(10, 6))

for label, color, file in zip(labels, colors, files):
    jet_pt = file.root['jet_pt'][:]
    weights = file.root['weight'][:]    
    plt.scatter(jet_pt, 
                weights, 
                alpha=0.2,
                color=color,
                s=1, 
                label=label.replace('label_', ''))
plt.legend()  
plt.xlabel('jet_pt')
plt.ylabel('weight')
plt.title('jet_pt vs weight distribution ')
plt.grid(True)

plt.savefig(f'{output_path}/jet_pt_vs_weight.png')
plt.close()

file.close()