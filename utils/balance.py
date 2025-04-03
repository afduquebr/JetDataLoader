import tables
import numpy as np
import time

# Path
path = "/AtlasDisk/user/duquebran/JetTagging/5-classes/data_train"

# Files
input_files = [
    f"{path}/train_files/train_QCD.h5",
    f"{path}/train_files/train_W.h5",
    f"{path}/train_files/train_Z.h5",
    f"{path}/train_files/train_top.h5",
    f"{path}/train_files/train_higgs.h5"
]

output_file = f"{path}/train_5M.h5"

# labels = ['label_QCD', 'label_WZ', 'label_top', 'label_higgs']
labels = ['label_QCD', 'label_W', 'label_Z', 'label_top', 'label_higgs']
chunk_size = 1000000  # Define a chunk size suitable for your memory constraints

print("Opening input HDF5 files...")
files = [tables.open_file(input_file, mode="r") for input_file in input_files]
print("Files opened successfully!")

# Get the indices where each label is 1
print("Extracting indices for each label...")
idx = [np.where(file.root[label][:] == 1)[0] for file, label in zip(files, labels)]

# Find the minimum number of samples across all labels
# num_samples = min(id.size for id in idx)
num_samples = 1000000
num_total = num_samples * len(labels)
print(f"Selected {num_samples} samples per class, total {num_total} samples.")

# Sample indices from each class
print("Sampling random indices...")
random_indices = np.array([np.random.choice(id, num_samples, replace=False) for id in idx])
print("Random indices sampled.")

num_chunks = int(np.ceil(num_samples / chunk_size))
chunk_shape = chunk_size * len(labels)
print(f"Processing in {num_chunks} chunks of size {chunk_shape}.")

# Create random indices for each chunk
chunks = [np.arange(chunk_shape) if (i + 1) * chunk_shape <= num_total 
          else np.arange(num_total - i * chunk_shape) 
          for i in range(num_chunks)]

# Shuffle each chunk independently
for chunk in chunks:
    np.random.shuffle(chunk)

# Open the output HDF5 file
with tables.open_file(output_file, mode="w") as outfile:
    print(f"Creating output file: {output_file}")

    for key in files[0].root._v_children:  # Loop through datasets
        print(f"Processing dataset: {key}")
        dataset_shape = (num_total,) + files[0].root[key].shape[1:]
        atom = tables.Atom.from_dtype(files[0].root[key].dtype)

        # Create the output dataset
        output_array = outfile.create_carray(
            outfile.root, 
            key, 
            atom, 
            dataset_shape,
            chunkshape=(chunk_shape,) + files[0].root[key].shape[1:]
        )
        print(f"Created dataset '{key}' with shape {dataset_shape}")

        # Process data in randomly split chunks
        for i, chunk in enumerate(chunks):
            start_time = time.time()
            start = i * chunk_size
            stop = min((i + 1) * chunk_size, num_samples)
            print(f"Processing indices {start}-{stop}")

            # Select random chunk indices
            chunk_indices = np.sort(random_indices[:, start:stop])

            # Extract the data for the chunk
            chunk_data = [file.root[key][:][chunk_id] for file, chunk_id in zip(files, chunk_indices)]

            # Save the chunk to the output dataset
            output_array[(start * len(labels)):(stop * len(labels))] = np.concatenate(chunk_data)[chunk]

            elapsed_time = time.time() - start_time
            print(f"Processed chunk {i+1}/{num_chunks} for '{key}' in {elapsed_time:.2f} sec.")

    print(f"Finished processing. Output file saved at: {output_file}")

# Close input files
for file in files:
    file.close()
print("All input files closed. Script completed successfully!")