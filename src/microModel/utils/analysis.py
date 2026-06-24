import os
import json
import pickle
import numpy as np
import torch
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report


def _init_plotting():
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"]


def extract_embeddings(model_fn, loader, device, get_inputs_fn):
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = get_inputs_fn(batch)
            emb = model_fn(inputs.to(device))
            embeddings.append(emb.cpu().numpy())
            if labels is not None:
                labels_list.append(labels)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    return embeddings, labels


def sample_balanced(embeddings, labels, max_samples):
    if labels is None or len(embeddings) <= max_samples:
        return embeddings, labels

    num_classes = len(set(int(l) for l in labels))
    per_class = max_samples // num_classes

    sampled_emb = []
    sampled_lbl = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) > per_class:
            idx = np.random.choice(idx, per_class, replace=False)
        sampled_emb.append(embeddings[idx])
        sampled_lbl.append(labels[idx])

    return np.concatenate(sampled_emb, axis=0), np.concatenate(sampled_lbl, axis=0)


def fit_and_save_umap(embeddings, labels, config, run_dir, class_names=None):
    _init_plotting()
    import matplotlib.pyplot as plt

    try:
        import warnings
        import umap
        from umap import UMAP
    except ImportError:
        print("umap-learn not installed. Skipping UMAP.")
        return None

    if len(embeddings) < 10:
        print(f"Skipping UMAP: too few samples ({len(embeddings)})")
        return None

    warnings.filterwarnings("ignore", category=UserWarning, module="umap")
    reducer = UMAP(
        n_components=2,
        n_neighbors=config["analysis"].get("umap_n_neighbors", 15),
        min_dist=config["analysis"].get("umap_min_dist", 0.1),
        random_state=0
    )
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        num_classes = len(set(int(l) for l in labels))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for c in range(num_classes):
            mask = labels == c
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[colors[c]],
                        label=class_names[c] if class_names else str(c), alpha=0.7, s=10)
        plt.legend()
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7, s=10, c="#4682B4")
    plt.title("UMAP (val set)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "umap_val.pdf"))
    plt.close()

    with open(os.path.join(run_dir, "umap_reducer.pkl"), "wb") as f:
        pickle.dump(reducer, f)
    print(f"  UMAP plot: umap_val.pdf")
    print(f"  UMAP reducer saved to {os.path.join(run_dir, 'umap_reducer.pkl')}")
    return reducer


def run_sl_analysis(model, val_loader, label_map, config, run_dir, device,
                    best_epoch=None, best_val_f1_macro=None):
    _init_plotting()
    import matplotlib.pyplot as plt

    extract_fn = lambda batch: (batch[0].to(device), batch[1].numpy())
    model_fn = lambda inputs: model.forward_head(model.forward_features(inputs), pre_logits=True)

    embeddings, labels = extract_embeddings(model_fn, val_loader, device, extract_fn)
    embeddings, labels = sample_balanced(
        embeddings, labels, config["analysis"]["max_samples"]
    )
    np.save(os.path.join(run_dir, "val_embeddings.npy"), embeddings)
    num_classes = len(label_map)
    idx_to_class = {v: k for k, v in label_map.items()}
    class_names_umap = [idx_to_class[i] for i in range(num_classes)]
    fit_and_save_umap(embeddings, labels, config, run_dir, class_names=class_names_umap)

    model.eval()
    val_all_preds = []
    val_all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            logits = model(inputs.to(device))
            val_all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            val_all_targets.append(targets.numpy())

    val_all_preds = np.concatenate(val_all_preds)
    val_all_targets = np.concatenate(val_all_targets)
    num_classes = len(label_map)
    idx_to_class = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    cm = confusion_matrix(val_all_targets, val_all_preds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, cm_data, title, fmt in [
        (ax1, cm, "Confusion Matrix", "d"),
        (ax2, cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10), "Confusion Matrix (Normalized)", ".2f"),
    ]:
        im = ax.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        thresh = cm_data.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm_data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_data[i, j] > thresh else "black")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.6)
    plt.savefig(os.path.join(run_dir, "confusion_matrix.pdf"))
    plt.close()

    report = classification_report(val_all_targets, val_all_preds,
                                   target_names=class_names, output_dict=True, zero_division=0)
    metrics = {"classification_report": report}
    if best_epoch is not None:
        metrics["best_epoch"] = best_epoch
        metrics["best_val_f1_macro"] = best_val_f1_macro
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Confusion matrix: confusion_matrix.pdf")
    print(f"  Metrics: metrics.json")


def run_ssl_analysis(backbone, head_fn, val_loader, config, run_dir, device, num_global):
    extract_fn = lambda batch: (torch.cat(batch["views"][:num_global], dim=0).to(device), None)
    model_fn = lambda inputs: head_fn(backbone.forward_features(inputs))

    embeddings, _ = extract_embeddings(model_fn, val_loader, device, extract_fn)
    np.save(os.path.join(run_dir, "val_embeddings.npy"), embeddings)
    fit_and_save_umap(embeddings, None, config, run_dir)
