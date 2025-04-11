import tables
import numpy as np
import time
import glob

def balance(input_path, mode, labels, output_filename, size_per_class=None, chunk_size=1000000):
    """
    Process multiple HDF5 files by sampling an equal number of entries per class and saving them to a new file.

    Args:
        input_path (str): Path to the directory containing input HDF5 files.
        mode (str): Mode of operation, e.g., "train" or "test".
        labels (list): List of label names corresponding to the classes.
        output_filename (str): Path for the output HDF5 file.
        size_per_class (int): Number of samples to select per class.
        chunk_size (int): Size of chunks to process at a time.

    Returns:
        None
    """

    # Input files based on mode
    print("Opening input HDF5 files...")
    files = [tables.open_file(input_file, mode="r") for input_file in glob.glob(f"{input_path}/{mode}_files/*.h5")]
    print("Files opened successfully!")

    # Get indices where each label is 1
    print("Extracting indices for each label...")
    idx = [np.where(file.root[label][:] == 1)[0] for file, label in zip(files, labels)]

    # Number of samples per class (can be dynamic, but we use user-defined)
    if size_per_class is None:
        size_per_class = min(id.size for id in idx)

    num_total = size_per_class * len(labels)
    print(f"Selected {size_per_class} samples per class, total {num_total} samples.")

    # Sample random indices from each class
    print("Sampling random indices...")
    random_indices = np.array([
        np.random.choice(class_idx, size_per_class, replace=False) 
        for class_idx in idx
    ])
    print("Random indices sampled.")

    num_chunks = int(np.ceil(size_per_class / chunk_size))
    chunk_shape = chunk_size * len(labels)
    print(f"Processing in {num_chunks} chunks of size {chunk_shape}.")

    # Prepare chunks
    chunks = [
        np.arange(chunk_shape) if (i + 1) * chunk_shape <= num_total
        else np.arange(num_total - i * chunk_shape)
        for i in range(num_chunks)
    ]

    # Shuffle each chunk independently
    for chunk in chunks:
        np.random.shuffle(chunk)

    # Open output file
    with tables.open_file(output_filename, mode="w") as outfile:
        print(f"Creating output file: {output_filename}")

        # Iterate over datasets in the first input file
        for key in files[0].root._v_children:
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

            # Process in chunks
            for i, chunk in enumerate(chunks):
                start_time = time.time()
                start = i * chunk_size
                stop = min((i + 1) * chunk_size, size_per_class)
                print(f"Processing indices {start}-{stop}")

                # Select indices for the chunk
                chunk_indices = np.sort(random_indices[:, start:stop])

                # Initialize buffer for data
                data_buffer = np.concatenate([
                    file.root[key][indices]
                    for file, indices in zip(files, chunk_indices)
                ])

                # Write the data buffer to the output file
                output_array[chunk] = data_buffer

                elapsed_time = time.time() - start_time
                print(f"Chunk {i+1}/{num_chunks} processed in {elapsed_time:.2f} seconds.")

    # Close input files
    for file in files:
        file.close()

    print(f"Data processing completed successfully! Output saved to {output_filename}")


if __name__ == "__main__":

    num_classes = 4

    path = f"/sps/atlas/a/aduque/JetTagging/{num_classes}-classes/data_train"

    if num_classes == 4:
        labels = ['label_QCD', 'label_WZ', 'label_top', 'label_higgs']
    elif num_classes == 5:
        labels = ['label_QCD', 'label_W', 'label_Z', 'label_top', 'label_higgs']

    chunk_size = 1000000  # Define a chunk size suitable for your memory constraints

    balance(
        input_path=path,
        mode='train',
        labels=labels,
        output_filename="{path}/test_4M.h5",
        size_per_class=1000000,
        chunk_size=chunk_size 
    )

    balance(
        input_path=path,
        mode='test',
        labels=labels,
        output_filename=f"{path}/test_1M.h5",
        size_per_class=250000,
        chunk_size=chunk_size 
    )

