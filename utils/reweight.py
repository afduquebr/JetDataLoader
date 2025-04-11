import tables
import numpy as np
import glob
import os
from reweightClass import CustomReweighter


def match_weights(source, target, n_bins=400):
    """ Calculate weights to match pt distribution to the target distribution. """
    reweighter = CustomReweighter(n_bins=n_bins, log_bins=True)
    reweighter.fit(source, target)
    # print(f"Reweighter edges: {np.power(10, reweighter.bin_edges)}")
    print(f"Reweighter edges: {reweighter.bin_edges}")
    weights = reweighter.predict_weights(source)
    return weights  

def process_h5_file(h5_file, data_source, dataset_weights=None, weight_start=0):
    """ Create or append datasets in an HDF5 file. """
    weight_idx = weight_start  # Track where we left off in weights
    
    for key in data_source.root._v_children:
        print(f"Processing dataset: {key}")

        if key not in h5_file.root:
            atom = tables.Atom.from_dtype(data_source.root[key].dtype)
            shape = (0,) + data_source.root[key][:].shape[1:] if isinstance(data_source.root[key][:][0], np.ndarray) else (0,)
            dataset = h5_file.create_earray(h5_file.root, key, atom, shape)
        else:
            dataset = getattr(h5_file.root, key)

        # Handle weight reweighting
        if key == "weight" and dataset_weights is not None:
            n_data = data_source.root["weight"][:].size
            dataset.append(dataset_weights[weight_idx : weight_idx + n_data])
            weight_idx += n_data  # Update weight index
        else:
            dataset.append(data_source.root[key][:])

    return weight_idx  # Return the updated weight index

def process_files(file_dir, out_dir, sig_tag, bkg_tag):
    """ Process signal and background HDF5 files, reweight, and save. """
    
    print(f"Processing files from bkgs: {bkg_tag} and sigs: {sig_tag}")

    sig_files = [tables.open_file(f, mode="r") for f in glob.glob(f"{file_dir}/{sig_tag}*.h5")]
    bkg_files = [tables.open_file(f, mode="r") for f in glob.glob(f"{file_dir}/{bkg_tag}*.h5")]

    # Process training data for reweighting
    print("Computing reweighting factors...")
    train_sig_pt, train_bkg_pt = [], []
    
    for sig_f, bkg_f, sig in zip(sig_files, bkg_files, glob.glob(f"{file_dir}/{sig_tag}*.h5")):
        if "train" in sig:
            train_sig_pt.append(sig_f.root["jet_pt"][:])
            train_bkg_pt.append(bkg_f.root["jet_pt"][:])

    train_sig_pt = np.concatenate(train_sig_pt)
    train_bkg_pt = np.concatenate(train_bkg_pt)
    bkg_weights = match_weights(train_bkg_pt, train_sig_pt)

    # Process background files
    print("Reweighting and saving background files...")
    os.makedirs(f"{out_dir}/train_files", exist_ok=True)
    os.makedirs(f"{out_dir}/test_files", exist_ok=True)

    train_bkg_file = tables.open_file(f"{out_dir}/train_files/train_{bkg_tag}.h5", mode="w")
    test_bkg_file = tables.open_file(f"{out_dir}/test_files/test_{bkg_tag}.h5", mode="w")

    weight_idx = 0
    for bkg_f, bkg_name in zip(bkg_files, glob.glob(f"{file_dir}/{bkg_tag}*.h5")):
        if "train" in bkg_name:
            weight_idx = process_h5_file(train_bkg_file, bkg_f, dataset_weights=bkg_weights, weight_start=weight_idx)
        else:
            process_h5_file(test_bkg_file, bkg_f)
    
    train_bkg_file.close()
    test_bkg_file.close()
    print("Completed background file processing.")

    # Process signal files

    train_sig_path = f"{out_dir}/train_files/train_{sig_tag}.h5"
    test_sig_path = f"{out_dir}/test_files/test_{sig_tag}.h5"
    
    if os.path.exists(train_sig_path) and os.path.exists(test_sig_path):
        print("Signal files already exist. Skipping creation.")
        return
    
    print("Merging signal files into unified train/test files...")
    train_sig_file = tables.open_file(train_sig_path, mode="w")
    test_sig_file = tables.open_file(test_sig_path, mode="w")

    for sig_f, sig_name in zip(sig_files, glob.glob(f"{file_dir}/{sig_tag}*.h5")):
        if "train" in sig_name:
            process_h5_file(train_sig_file, sig_f)
        else:
            process_h5_file(test_sig_file, sig_f)

    train_sig_file.close()
    test_sig_file.close()
    print("Completed signal file processing.")
    print("Done!")


if __name__ == "__main__":    
    input_dir = "/sps/atlas/a/aduque/JetTagging/4-classes/data_out"
    output_dir = "/sps/atlas/a/aduque/JetTagging/4-classes/data_train_new_reweight_log"
    process_files(input_dir, output_dir, "higgs", "QCD")
    process_files(input_dir, output_dir, "higgs", "top")
    process_files(input_dir, output_dir, "higgs", "WZ")