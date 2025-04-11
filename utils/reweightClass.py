import numpy as np
from sklearn.neighbors import KernelDensity

class CustomReweighter:
    def __init__(self, n_bins=400, log_bins=True, bandwidth=0.2, switch_value=None):
        self.n_bins = n_bins
        self.log_bins = log_bins
        self.bandwidth = bandwidth
        self.switch_value = switch_value
        self.bin_edges = None
        self.hist_source = None
        self.hist_target = None

    def _binning(self, source, target):
        # Manual log binning over full range of source and target
        min_edge = max(1e-3, min(source.min(), target.min()))
        max_edge = max(source.max(), target.max())
        if self.log_bins:
            bins = np.logspace(np.log10(min_edge), np.log10(max_edge), self.n_bins + 1)
        else:
            bins = np.linspace(min_edge, max_edge, self.n_bins + 1)
        return bins

    def fit(self, source, target):
        source = np.asarray(source).flatten()
        target = np.asarray(target).flatten()

        self.bin_edges = self._binning(source, target)

        self.hist_source, _ = np.histogram(source, bins=self.bin_edges)
        self.hist_target, _ = np.histogram(target, bins=self.bin_edges)

        # Avoid zero division
        self.bin_weights = np.zeros_like(self.hist_source, dtype=float)
        mask = self.hist_source > 0
        self.bin_weights[mask] = self.hist_target[mask] / self.hist_source[mask]

        # Normalize to mean weight = 1
        mean_weight = np.average(self.bin_weights[mask], weights=self.hist_source[mask])
        self.bin_weights /= mean_weight
 
        # Prepare KDE for tail region
        if self.switch_value is not None:
            mask_high = source > self.switch_value
            if np.any(mask_high):
                self.kde_source = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(source[mask_high].reshape(-1, 1))
                self.kde_target = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(target.reshape(-1, 1))
            else:
                self.kde_source = None
                self.kde_target = None

    def predict_weights(self, source):
        source = np.asarray(source).flatten()
        weights = np.ones_like(source, dtype=float)

        # Assign weights based on binning
        bin_indices = np.searchsorted(self.bin_edges, source, side='right') - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < len(self.bin_weights))
        weights[valid_bins] = self.bin_weights[bin_indices[valid_bins]]

        # Apply KDE smoothing for tail region
        if self.switch_value is not None:
            mask_high = source > self.switch_value
            if self.kde_source is not None and np.any(mask_high):
                source_high = source[mask_high].reshape(-1, 1)
                log_source_density = self.kde_source.score_samples(source_high)
                log_target_density = self.kde_target.score_samples(source_high)
                kde_weights = np.exp(log_target_density - log_source_density)
                kde_weights /= np.mean(kde_weights)  # normalize tail weights
                weights[mask_high] = kde_weights

        # Final normalization
        weights /= np.mean(weights)

        return weights
