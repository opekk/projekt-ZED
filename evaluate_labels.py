import os
import pandas as pd

LABELED_DIR = 'artifacts/labeled_datasets'
FEATURE_SETS = ['top29', 'top15', 'top10']
P_VALUES = [500, 1000, 1500]


def evaluate_file(path):
    df = pd.read_csv(path, usecols=['Class', 'generated_class'])
    y_true = df['Class']
    y_pred = df['generated_class']

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'flagged': tp + fp, 'total_frauds': tp + fn,
    }


def main():
    print("--- Ewaluacja wygenerowanych etykiet vs ground truth ---\n")
    rows = []

    for fs in FEATURE_SETS:
        for p in P_VALUES:
            path = os.path.join(LABELED_DIR, fs, f'labeled_data_p{p}.csv')
            if not os.path.exists(path):
                print(f"[POMINIĘTO] Brak pliku: {path}")
                continue

            metrics = evaluate_file(path)
            metrics['feature_set'] = fs
            metrics['P'] = p
            rows.append(metrics)

            print(f"[{fs} / p={p}]  flagged={metrics['flagged']}  "
                  f"TP={metrics['TP']}  FP={metrics['FP']}  "
                  f"FN={metrics['FN']}  TN={metrics['TN']}  "
                  f"prec={metrics['precision']:.4f}  "
                  f"rec={metrics['recall']:.4f}  "
                  f"f1={metrics['f1']:.4f}")

    if not rows:
        print("Brak plików do ewaluacji.")
        return

    summary = pd.DataFrame(rows)[
        ['feature_set', 'P', 'flagged', 'TP', 'FP', 'FN', 'TN',
         'precision', 'recall', 'f1']
    ]

    out_path = os.path.join('artifacts', 'label_evaluation.csv')
    summary.to_csv(out_path, index=False)
    print(f"\nPodsumowanie zapisane do: {out_path}")

    print("\n--- Recall grid (TP / total frauds) ---")
    print(summary.pivot(index='feature_set', columns='P', values='TP'))

    print("\n--- Precision grid ---")
    print(summary.pivot(index='feature_set', columns='P', values='precision').round(4))


if __name__ == '__main__':
    main()
