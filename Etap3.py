import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score

# Flaga konfiguracyjna - True = smoke test, False = pełna ewaluacja
SMOKE_TEST = False
# Eksperyment: jeśli True, ewaluuje przeciwko generated_class zamiast Class.
# To pomaga sprawdzić hipotezę, że paper używa AE-etykiet także do ewaluacji
# (co dałoby znacznie wyższe AUPRC). False = poprawna metodologia (ground truth).
EVAL_AGAINST_GENERATED = False

LABELED_DIR = 'artifacts/labeled_datasets'
RANKINGS_PATH = 'artifacts/feature_rankings.json'

FEATURE_SETS_FULL = ['top29', 'top15', 'top10']
P_VALUES_FULL = [500, 1000, 1500]

FEATURE_SETS_SMOKE = ['top15']
P_VALUES_SMOKE = [1500]

N_REPEATS_FULL = 10
N_REPEATS_SMOKE = 1
N_SPLITS = 5


def build_models():
    return {
        'DT': DecisionTreeClassifier(random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42),
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), batch_size=200,
                             max_iter=300, random_state=42),
    }


def feature_set_to_key(feature_set_name):
    """top15 -> top_15 (klucz w feature_rankings.json)"""
    return feature_set_name.replace('top', 'top_')


def evaluate_fold(X_train, X_test, y_gen_train, y_true_test):
    """Ewaluuje wszystkie 5 modeli na jednym foldzie. Zwraca dict model -> AUPRC."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # 4 modele nadzorowane - uczone na y_gen_train, oceniane na y_true_test
    for name, model in build_models().items():
        model.fit(X_train_s, y_gen_train)
        proba = model.predict_proba(X_test_s)[:, 1]
        results[name] = average_precision_score(y_true_test, proba)

    # Isolation Forest - nienadzorowany baseline (nie widzi y_gen_train)
    iso = IsolationForest(n_estimators=100, contamination='auto',
                          random_state=42, n_jobs=-1)
    iso.fit(X_train_s)
    scores = -iso.score_samples(X_test_s)  # negacja: wyższe = bardziej anomalia
    results['IF'] = average_precision_score(y_true_test, scores)

    return results


def evaluate_dataset(feature_set, p, features, n_repeats):
    """Ewaluacja jednego z 9 zbiorów. Zwraca listę wierszy do zapisu."""
    csv_path = os.path.join(LABELED_DIR, feature_set, f'labeled_data_p{p}.csv')
    print(f"\n=== {feature_set} / p={p} ===")
    print(f"Wczytywanie: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df[features].values
    y_generated = df['generated_class'].values
    y_true = df['Class'].values

    rows = []
    for repeat in range(n_repeats):
        seed = 42 + repeat
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        # Stratyfikacja na y_true (ground truth) - zgodnie z paperem, str. 14:
        # "the ground truth labels are used for the splits"
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_true)):
            t0 = time.time()
            y_test_eval = y_generated[test_idx] if EVAL_AGAINST_GENERATED else y_true[test_idx]
            fold_results = evaluate_fold(
                X[train_idx], X[test_idx],
                y_generated[train_idx], y_test_eval,
            )
            elapsed = time.time() - t0

            n_train_pos = int(y_generated[train_idx].sum())
            n_test_true_pos = int(y_true[test_idx].sum())

            for model_name, auprc in fold_results.items():
                rows.append({
                    'feature_set': feature_set,
                    'P': p,
                    'model': model_name,
                    'repeat': repeat,
                    'fold': fold,
                    'auprc': auprc,
                    'n_train_positives': n_train_pos,
                })

            scores_str = '  '.join(f'{m}={v:.4f}' for m, v in fold_results.items())
            print(f"  repeat={repeat} fold={fold} "
                  f"(train_pos={n_train_pos}, test_true_pos={n_test_true_pos}, "
                  f"{elapsed:.1f}s)  {scores_str}")

    return rows


def main():
    mode = "SMOKE TEST" if SMOKE_TEST else "PEŁNA EWALUACJA"
    eval_target = "generated_class (AE-etykiety)" if EVAL_AGAINST_GENERATED else "Class (ground truth)"
    print(f"--- ETAP 3: {mode} ---")
    print(f"Ewaluacja przeciwko: {eval_target}")

    feature_sets = FEATURE_SETS_SMOKE if SMOKE_TEST else FEATURE_SETS_FULL
    p_values = P_VALUES_SMOKE if SMOKE_TEST else P_VALUES_FULL
    n_repeats = N_REPEATS_SMOKE if SMOKE_TEST else N_REPEATS_FULL

    expected_runs = len(feature_sets) * len(p_values) * 5 * n_repeats * N_SPLITS
    print(f"Planowane uruchomienia: {len(feature_sets)} dataset × "
          f"{len(p_values)} P × 5 modeli × {n_repeats} powtórzeń × "
          f"{N_SPLITS} foldów = {expected_runs}")

    with open(RANKINGS_PATH, 'r') as f:
        feature_rankings = json.load(f)

    all_rows = []
    t_start = time.time()

    for fs in feature_sets:
        features = feature_rankings[feature_set_to_key(fs)]
        for p in p_values:
            all_rows.extend(evaluate_dataset(fs, p, features, n_repeats))

    total_elapsed = time.time() - t_start
    print(f"\n--- Zakończono w {total_elapsed:.1f}s ---")

    out_name = 'etap3_results_smoke.csv' if SMOKE_TEST else 'etap3_results.csv'
    out_path = os.path.join('artifacts', out_name)
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_path, index=False)
    print(f"Wyniki zapisane do: {out_path}  ({len(results_df)} wierszy)")

    print("\n--- Średnie AUPRC (feature_set × P × model) ---")
    summary = results_df.groupby(['feature_set', 'P', 'model'])['auprc'].agg(['mean', 'std'])
    print(summary.round(4))


if __name__ == '__main__':
    main()
