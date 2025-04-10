import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary

import tables
import numpy as np
import os
import yaml

path = "/AtlasDisk/user/duquebran/JetTagging/5-classes/data_train/"
data_config = yaml.safe_load(open('config/train_config.yaml', 'r'))

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, path, data_config, batch_size=128, num_workers=1):
        
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_data(self):
        train_dataset = self._load_dataset(self.data_config['train'])
        test_dataset = self._load_dataset(self.data_config['test'])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader



#########################################

import numpy as np
import tables
import os
from glob import glob
from sklearn.utils import shuffle

class DataLoader:
    def __init__(self, data_dir, batch_size=32, mode='train', config=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.config = config

        self.files = self._get_files()
        self.data = self._load_data()
        self.num_samples = len(next(iter(self.data.values())))
        self.current_idx = 0

    def _get_files(self):
        pattern = os.path.join(self.data_dir, f'*{self.mode}*.h5')
        return glob(pattern)

    def _load_data(self):
        data = {}
        for file_path in self.files:
            with tables.open_file(file_path, mode='r') as f:
                for array in f.root._f_list_nodes():
                    name = array.name
                    if name not in data:
                        data[name] = []
                    data[name].append(array.read())

        # Concatenate arrays from different files
        for key in data:
            data[key] = np.concatenate(data[key], axis=0)

        # Shuffle data
        data = shuffle(data, random_state=42)

        return data

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        batch_slice = slice(self.current_idx, self.current_idx + self.batch_size)
        batch = {key: value[batch_slice] for key, value in self.data.items()}
        self.current_idx += self.batch_size

        return batch

    def get_input_output(self, batch):
        # Extract input and label from batch based on config
        inputs = {}

        for input_name, input_cfg in self.config['inputs'].items():
            inputs[input_name] = []
            for var_cfg in input_cfg['vars']:
                var_name = var_cfg[0]
                inputs[input_name].append(batch[var_name])
            inputs[input_name] = np.stack(inputs[input_name], axis=-1)

        labels = np.stack([batch[label] for label in self.config['labels']['value']], axis=-1)

        weights = None
        if 'weights' in self.config and 'weight_names' in self.config['weights']:
            weights = np.stack([batch[weight_name] for weight_name in self.config['weights']['weight_names']], axis=-1)

        return inputs, labels, weights


# Example usage:
# config = load_yaml('config.yaml')
# data_loader = DataLoader(data_dir='path/to/h5/files', batch_size=64, mode='train', config=config)
# for batch in data_loader:
#     inputs, labels, weights = data_loader.get_input_output(batch)
#     model.train_on_batch(inputs, labels, sample_weight=weights)




import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tables
import glob
import os


class H5Dataset(Dataset):
    def __init__(self, file_list, config, transform=None):
        self.file_list = file_list
        self.config = config
        self.transform = transform
        
        # Gather all indices
        self.data_index = []
        self.file_handles = []

        for file_path in self.file_list:
            f = tables.open_file(file_path, mode='r')
            n_events = f.root.label_QCD.shape[0]  # assuming all labels have same length
            self.data_index.extend([(len(self.file_handles), i) for i in range(n_events)])
            self.file_handles.append(f)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_idx, row_idx = self.data_index[idx]
        f = self.file_handles[file_idx]

        # === Prepare inputs === #
        inputs = {}
        for input_name, input_cfg in self.config['inputs'].items():
            arrays = []
            for var in input_cfg['vars']:
                var_name = var[0] if isinstance(var, list) else var
                array = f.root[var_name][row_idx]
                arrays.append(array)
            inputs[input_name] = np.stack(arrays, axis=-1)  # shape: (length, features)

        # === Prepare mask === #
        masks = []
        for var in self.config['pf_mask']['vars']:
            var_name = var[0] if isinstance(var, list) else var
            mask = f.root[var_name][row_idx]
            masks.append(mask)
        inputs['pf_mask'] = np.stack(masks, axis=-1)

        # === Prepare labels === #
        labels = []
        for label_name in self.config['labels']['value']:
            labels.append(f.root[label_name][row_idx])
        labels = np.stack(labels).astype(np.float32)

        # === Prepare weights === #
        weights = []
        for weight_name in self.config['weights']['weight_names']:
            weights.append(f.root[weight_name][row_idx])
        weights = np.stack(weights).astype(np.float32)

        # === Transform to tensors === #
        inputs = {k: torch.tensor(v, dtype=torch.float32) for k, v in inputs.items()}
        labels = torch.tensor(labels, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        if self.transform:
            inputs = self.transform(inputs)

        return inputs, labels, weights

    def close(self):
        for f in self.file_handles:
            f.close()


# === Helper function to create dataloader === #
def create_dataloader(data_dir, config, batch_size=32, shuffle=True, mode='train', num_workers=4):
    file_pattern = os.path.join(data_dir, f"*{mode}*.h5")
    file_list = glob.glob(file_pattern)
    dataset = H5Dataset(file_list, config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader, dataset


# === Example usage === #
# config = load_yaml_config("config.yaml")
# train_loader, train_dataset = create_dataloader("/path/to/data", config, batch_size=64, mode='train')

# for batch_inputs, batch_labels, batch_weights in train_loader:
#     # feed to model


import os
import tables
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class H5Dataset(Dataset):
    def __init__(self, data_dir, mode='train', config=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if mode in f and f.endswith('.h5')]
        if not self.files:
            raise ValueError(f"No H5 files found in {data_dir} for mode '{mode}'")
        self.config = config or {}
        self.data = self.load_data()

    def load_data(self):
        data = {'inputs': {}, 'labels': {}, 'weights': []}
        
        input_keys = list(self.config['inputs'].keys())
        label_keys = self.config['labels']['value']
        weight_keys = self.config['weights']['weight_names']

        for file_path in self.files:
            with tables.open_file(file_path, mode='r') as f:
                for input_name in input_keys:
                    input_vars = self.config['inputs'][input_name]['vars']
                    

                    input_data = [getattr(f.root, var[0])[:] for var in input_vars]
                    input_data = np.stack(input_data, axis=-1)  # shape: (events, particles, features)

                    if input_name not in data['inputs']:
                        data['inputs'][input_name] = [input_data]
                    else:
                        data['inputs'][input_name].append(input_data)

                labels = [getattr(f.root, label)[:] for label in label_keys]
                weights = [getattr(f.root, weight)[:] for weight in weight_keys]

                labels = np.stack(labels, axis=-1)
                weights = np.stack(weights, axis=-1)

                if 'labels' not in data:
                    data['labels'] = [labels]
                else:
                    data['labels'].append(labels)

                data['weights'].append(weights)

        # Concatenate data from all files
        for key in data['inputs']:
            data['inputs'][key] = np.concatenate(data['inputs'][key], axis=0)
        data['labels'] = np.concatenate(data['labels'], axis=0)
        data['weights'] = np.concatenate(data['weights'], axis=0)

        return data

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        inputs = {key: torch.tensor(self.data['inputs'][key][idx], dtype=torch.float32) for key in self.data['inputs']}
        labels = torch.tensor(self.data['labels'][idx], dtype=torch.float32)
        weights = torch.tensor(self.data['weights'][idx], dtype=torch.float32)
        return inputs, labels, weights


def create_dataloader(data_dir, config, mode='train', batch_size=1024, num_workers=4, shuffle=True):
    dataset = H5Dataset(data_dir=data_dir, mode=mode, config=config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader
