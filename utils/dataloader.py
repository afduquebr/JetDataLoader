import os
import tables
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml

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

if __name__ == "__main__":
    # Example usage
    data_dir = "/AtlasDisk/user/duquebran/JetTagging/5-classes/data_train/"
    data_config = yaml.safe_load(open('config/train_config.yaml', 'r'))
    batch_size = 1024
    num_workers = 1

    train_loader = create_dataloader(data_dir, data_config, "train", batch_size, num_workers)
    test_loader = create_dataloader(data_dir, data_config, "test", batch_size, num_workers)
