import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set nice plot style
sns.set(style="whitegrid", context="talk", palette="muted")

num_classes = 4

input_path = f"/sps/atlas/a/aduque/JetTagging/{num_classes}-classes/data_train/train_5M.h5"
output_path = f"/pbs/home/a/aduque/private/JetDataLoader/figs/{num_classes}-classes/input_data/"

os.makedirs(output_path, exist_ok=True)

# Class labels
if num_classes == 4:
    labels = ['label_QCD', 'label_WZ', 'label_top', 'label_higgs']
elif num_classes == 5:
    labels = ['label_QCD', 'label_W', 'label_Z', 'label_top', 'label_higgs']

# Open HDF5 file
file = tb.open_file(input_path, mode="r")

# Indices for each class
idx = [np.where(file.root[label][:] == 1)[0] for label in labels]

# Color palette
colors = sns.color_palette("muted", len(labels))

# # Loop over all features
# for feature in file.root._v_children:
#     data_sample = file.root[feature][:]

#     # Check if the variable is multi-dimensional (per-constituent)
#     is_multidim = len(data_sample.shape) > 1

#     # Skip empty arrays
#     if data_sample.size == 0:
#         continue

#     plt.figure(figsize=(12, 7))

#     use_weights = (feature == 'jet_pt')

#     for label, id, color in zip(labels, idx, colors):
#         data = data_sample[id]

#         # Flatten if multidimensional (e.g., (N_events, 80, 1) → (N_events * 80,))
#         if is_multidim:
#             data = data.reshape(-1)

#         weights = None
#         if use_weights:
#             weights = file.root['weight'][:][id]
#             if is_multidim:
#                 # Repeat weights per constituent
#                 weights = np.repeat(weights, data_sample.shape[1])

#         # Exclude data equal to -999
#         mask = data != -999
#         data = data[mask]
#         if weights is not None:
#             weights = weights[mask]

#         plt.hist(data,
#                  bins=50,
#                  alpha=0.4,
#                  label=label.replace('label_', ''),
#                  weights=weights,
#                  color=color,
#                  histtype='stepfilled',
#                  edgecolor='black',
#                  linewidth=1.2)

#     feature_name = feature.replace('_', ' ').title()

#     plt.xlabel(feature_name, fontsize=14)
#     plt.ylabel('Counts', fontsize=14)
#     plt.yscale('log')
#     if feature == 'weight':
#         plt.xscale('log')
#     plt.title(f'{feature_name} Distribution ({num_classes} Classes)', fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()

#     plt.savefig(f'{output_path}/{feature}_distribution.pdf', dpi=300, transparent=True)
#     plt.close()

# Step 1: Compute the count of valid clusters per event
clus_E = file.root['fjet_clus_E'][:]
valid_clus_mask = clus_E != -999
valid_clus_count = valid_clus_mask.sum(axis=1)  # shape: (n_events,)

# Step 2: For global variables (jet_pt, jet_E), plot wrt valid_clus_count
global_features = ['jet_pt', 'jet_energy']

for feature in global_features:

    for label, id in zip(labels, idx):
        feature_data = file.root[feature][:]

        plt.figure(figsize=(10, 6))
        plt.scatter(valid_clus_count[id], feature_data[id], alpha=0.2, s=1, label=label.replace('label_', ''))  # s=1 makes the scatter points small
        plt.xlabel('Number of Valid Clusters')
        plt.ylabel(feature)
        plt.legend()
        plt.title(f'{feature} vs Number of Valid Clusters')
        plt.grid(True)

        plt.savefig(f'{output_path}/{feature}_vs_valid_clusters.png')
        plt.close()

# Step: Plot jet_pt vs weight
jet_pt = file.root['jet_pt'][:]
weights = file.root['weight'][:]


plt.figure(figsize=(10, 6))

for label, id in zip(labels, idx):
    plt.scatter(jet_pt[id], weights[id], alpha=0.2, s=1, label=label.replace('label_', ''))
plt.legend()  
plt.xlabel('jet_pt')
plt.ylabel('weight')
plt.title('jet_pt vs weight distribution')
plt.grid(True)

plt.savefig(f'{output_path}/jet_pt_vs_weight.pdf')
plt.close()


file.close()

