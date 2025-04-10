import uproot as up
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def plot(sample, model, labels):
    # Define input and output paths
    input_path = os.path.join("training", "Pythia", "kin", model, "predict_output", f"pred_sample_{sample}.root")
    output_path = os.path.join("figs", model, sample)

    # Create output directory if not exists
    os.makedirs(output_path, exist_ok=True)

    try:
        # Open ROOT file
        with up.open(input_path) as file:
            if not file.keys():
                raise ValueError(f"Empty ROOT file: {input_path}")

            # Assuming only one tree
            tree_name = file.keys()[0]
            tree = file[tree_name]

            # Load data as a Pandas DataFrame
            data = tree.arrays(library="pd")

    except Exception as e:
        print(f"Error loading file {input_path}: {e}")
        return

    # Handling true labels and predictions
    tags = [f"label_{label}" for label in labels]
    score_tags = [f"score_{tag}" for tag in tags]

    # Convert boolean columns to `bool` type
    data[tags] = data[tags].replace({"True": True, "False": False}).astype(bool)

    # Extract true and predicted labels
    data["true_label"] = data[tags].idxmax(axis=1).str.split("_").str[1]
    data["predict_label"] = data[score_tags].idxmax(axis=1).str.split("_").str[2]

    # 1. Plot Confusion Matrix
    cm = confusion_matrix(data["true_label"], data["predict_label"], labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_path}/Confusion_Matrix.pdf")
    plt.clf()

    # 2. Distribution of DNN outputs for all classes
    plt.figure()
    for score_tag in score_tags:
        plt.hist(data[score_tag], bins=30, range=[0, 1], histtype='step', label=score_tag.split("_")[2])
    plt.xlabel("DNN output")
    plt.ylabel("Number of jets")
    plt.yscale('log')
    plt.title("Histograms of QCD, WZ, top, and H")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_path}/output_distribution.pdf")
    plt.clf()

    # 3. ROC Curve (Background rejection for each class)
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3))]
    for label in labels:
        plt.figure()
        for i, label2 in enumerate(labels):
            if label == label2:
                continue

            # Select only events of class `label` and `label2`
            mask = (data["true_label"] == label) | (data["true_label"] == label2)

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(data["true_label"][mask], data[f"score_label_{label}"][mask], pos_label=label)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_fpr = 1. / fpr
                inv_fpr[np.isinf(inv_fpr)] = 1e6  # Replace infinities for better plotting
            
            plt.plot(tpr, inv_fpr, label=label2, linestyle=linestyles[i])

        plt.xlim(0, 1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection (1/FPR)')
        plt.yscale('log')
        plt.legend()
        plt.title(f'ROC curve for {label} vs others')
        plt.savefig(f"{output_path}/{label}_ROC.pdf")
        plt.clf()

    # 4. Discriminant for each class
    for label, score_tag in zip(labels, score_tags):
        plt.figure()
        for label2 in labels:
            disc = data[score_tag][(data["true_label"] == label2)]
            plt.hist(disc, bins=40, range=(0, 1), histtype='step', label=label2)
        plt.xlabel('DNN output')
        plt.ylabel('Number of jets')
        plt.yscale('log')
        plt.title(f'Discriminant for {label}')
        plt.legend()
        plt.savefig(f"{output_path}/{label}_output_dist.pdf")
        plt.clf()

    print(f"Plots saved in {output_path}") 

    # 5. Background rejection vs pT for all classes
    # Convert pT to GeV
    data["fjet_pt_GeV"] = data["fjet_pt"] / 1000.0

    # Define signal efficiency
    signal_efficiency = 0.5

    # Define pT bins (GeV)
    pt_bins = np.linspace(300, 3000, 15)  # Adjust bins as needed
    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    
    plt.figure(figsize=(7,6))

    for label in labels:
        background_rejection = []
        rejection_err = []

        for j in range(len(pt_bins) - 1):
            pt_min, pt_max = pt_bins[j], pt_bins[j + 1]
            mask = (data["fjet_pt_GeV"] > pt_min) & (data["fjet_pt_GeV"] <= pt_max)

            if mask.sum() == 0:
                background_rejection.append(np.nan)
                rejection_err.append(np.nan)
                continue

            # Signal is the current class, background is everything else
            is_signal = (data[f"label_{label}"] == 1)[mask]
            scores = data[f"score_label_{label}"][mask]
            weights = data["test_w"][mask]

            try:
                fpr, tpr, _ = roc_curve(is_signal, scores, sample_weight=weights)
                rejection = 1.0 / (fpr + 1e-10)
                rejection[np.isinf(rejection)] = 1e6  # Avoid infinities

                # Find rejection at TPR ~ 50%
                idx = np.argmax(tpr >= signal_efficiency)
                rejection_value = rejection[idx] if np.any(tpr >= signal_efficiency) else np.nan

                background_rejection.append(rejection_value)
                rejection_err.append((pt_max - pt_min)/2)  # Example: 10% error (adjust if needed)

            except Exception:
                background_rejection.append(np.nan)
                rejection_err.append(np.nan)

        # Plot with error bars
        plt.errorbar(pt_centers, background_rejection, xerr=rejection_err, 
                     fmt="_", label=f"{label}", capsize=0, elinewidth=2.5)

    # ATLAS-like style
    plt.xscale("linear")
    plt.yscale("log")
    plt.xlim(300, 3000)
    plt.ylim(1, 1e6)
    plt.xlabel(r"Large-$R$ jet $p_T$ [GeV]", fontsize=14)
    plt.ylabel(r"Background rejection $(1 / \varepsilon_{\mathrm{bkg}}^{\mathrm{rel}})$", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.legend(fontsize=12, loc="best")
    plt.text(400, 4e5, r"$\bf{ATLAS}$ Simulation Preliminary", fontsize=14)
    plt.text(400, 2e5, r"$\sqrt{s} = 13$ TeV", fontsize=12)
    plt.text(400, 1e5, r"R = 1.0", fontsize=12)
    plt.text(400, 5e4, r"$\varepsilon_{\mathrm{sig}}^{\mathrm{rel}} = 50\%$, $|\eta| < 2.0$", fontsize=12)
    
    plt.savefig(f"{output_path}/Background_rejection_vs_pt.pdf", bbox_inches="tight")
    plt.clf()

    # 6. Background rejection vs pT per class at different signal efficiencies

    # Define signal efficiency
    signal_efficiency = [0.5, 0.8]

    # Define pT bins (GeV)
    pt_bins = np.linspace(300, 3000, 15)  # Adjust bins as needed
    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    for label in labels:

        plt.figure(figsize=(7,6))

        for signal_eff in signal_efficiency:

            background_rejection = []
            rejection_err = []

            for j in range(len(pt_bins) - 1):
                pt_min, pt_max = pt_bins[j], pt_bins[j + 1]
                mask = (data["fjet_pt_GeV"] > pt_min) & (data["fjet_pt_GeV"] <= pt_max)

                if mask.sum() == 0:
                    background_rejection.append(np.nan)
                    rejection_err.append(np.nan)
                    continue

                # Signal is the current class, background is everything else
                is_signal = (data[f"label_{label}"] == 1)[mask]
                scores = data[f"score_label_{label}"][mask]
                weights = data["test_w"][mask]

                try:
                    fpr, tpr, _ = roc_curve(is_signal, scores, sample_weight=weights)
                    rejection = 1.0 / (fpr + 1e-10)
                    rejection[np.isinf(rejection)] = 1e6  # Avoid infinities

                    # Find rejection at TPR ~ 50%
                    idx = np.argmax(tpr >= signal_eff)
                    rejection_value = rejection[idx] if np.any(tpr >= signal_eff) else np.nan

                    background_rejection.append(rejection_value)
                    rejection_err.append((pt_max - pt_min)/2)  # Example: 10% error (adjust if needed)

                except Exception:
                    background_rejection.append(np.nan)
                    rejection_err.append(np.nan)

            # Plot with error bars
            plt.errorbar(pt_centers, background_rejection, xerr=rejection_err, fmt="_", 
                        label=r"$\varepsilon_{\mathrm{sig}}^{\mathrm{rel}} = $" + f"{signal_eff * 100}%", capsize=0, elinewidth=2.5)

        # ATLAS-like style
        plt.xscale("linear")
        plt.yscale("log")
        plt.xlim(300, 3000)
        plt.ylim(1, 1e6)
        plt.xlabel(r"Large-$R$ jet $p_T$ [GeV]", fontsize=14)
        plt.ylabel(r"Background rejection $(1 / \varepsilon_{\mathrm{bkg}}^{\mathrm{rel}})$", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        plt.legend(fontsize=12, loc="best")
        plt.text(400, 4e5, r"$\bf{ATLAS}$ Simulation Preliminary", fontsize=14)
        plt.text(400, 2e5, r"$\sqrt{s} = 13$ TeV, " + f"{label} tagging", fontsize=12)
        plt.text(400, 1e5, r"R = 1.0", fontsize=12)

        plt.savefig(f"{output_path}/{label}_background_rejection_vs_pt.pdf", bbox_inches="tight")
        plt.clf()

    # 7. ROC curves (Background rejection for each class for different pT bins)
    # pt_bins = np.linspace(data["fjet_pt"].min(), data["fjet_pt"].max(), 11)  # 10 bins

    # for i, label in enumerate(labels):
    #     for pt_idx in range(len(pt_bins) - 1):
    #         pt_min = pt_bins[pt_idx]
    #         pt_max = pt_bins[pt_idx + 1]
    #         plt.figure()
    #         plt.yscale('log')

    #         # Ticks formatting for log scale
    #         locmaj = plt.LogLocator(base=10.0, subs=(0.1, 1.0,))
    #         plt.gca().yaxis.set_major_locator(locmaj)
    #         locmin = plt.LogLocator(base=10.0, subs=[0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10])
    #         plt.gca().yaxis.set_minor_locator(locmin)
    #         plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

    #         # Style
    #         for spine in ['bottom', 'top', 'right', 'left']:
    #             plt.gca().spines[spine].set_color('black')
    #         plt.tick_params(axis='x', colors='black')
    #         plt.tick_params(axis='y', colors='black')

    #         for j, label2 in enumerate(labels):
    #             if label == label2 or j == i:
    #                 continue

    #             # Mask for label pair and pt bin
    #             mask = ((data["true_label"] == label) | (data["true_label"] == label2)) & \
    #                     (data["fjet_pt"] > pt_min) & (data["fjet_pt"] < pt_max)

    #             if mask.sum() == 0:
    #                 continue

    #             fpr, tpr, _ = roc_curve(data["true_label"][mask], data[f"score_label_{label}"][mask], pos_label=label)

    #             with np.errstate(divide='ignore', invalid='ignore'):
    #                 inv_fpr = 1. / fpr
    #                 inv_fpr[np.isinf(inv_fpr)] = 1e6

    #             plt.plot(tpr, inv_fpr, label=label2, linestyle=linestyles[j % len(linestyles)])

    #         plt.xlabel('Signal efficiency')
    #         plt.ylabel('Background rejection (1/FPR)')
    #         plt.legend()
    #         plt.title(f'ROC: {label} vs others for {pt_min} < pT < {pt_max} GeV')
    #         plt.savefig(f"{output_path}/ROC_{label}_pt{pt_min}_{pt_max}.pdf", bbox_inches='tight')
    #         plt.clf()



if __name__ == "__main__":
    # Define the labels and samples
    # labels = ['QCD', 'WZ', 'top', 'higgs']
    labels = ['QCD', 'W', 'Z', 'top', 'higgs']
    models = ["ParT"]
    sample = "5M_5cl"

    # Run the plot function for each sample
    for model in models:
        plot(sample, model, labels)
