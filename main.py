import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
from scipy.stats import spearmanr

def overlap_at_k(ranking_a, ranking_b, k):
    """Zwraca liczbę wspólnych cech w top-k obu rankingów."""
    return len(set(ranking_a[:k]) & set(ranking_b[:k]))

def main():
    print("Wczytywanie danych...")
    df = pd.read_csv('creditcard.csv')
    
    # Przygotowanie cech: używamy tylko Amount oraz V1-V28
    features = [col for col in df.columns if col not in ['Time', 'Class']]
    X = df[features]
    
    print(f"Dane wczytane. Kształt: {X.shape}")

    # 1. Trening IF na całości danych
    print("Trenowanie Isolation Forest na pełnym zbiorze...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination='auto', 
        random_state=42, 
        n_jobs=-1
    )
    iso_forest.fit(X)

    # 2. Inicjalizacja SHAP
    print("Inicjalizacja SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(iso_forest)

    # 3. Inicjalizacja stałych Fazy A
    N_grid = [1000, 2000, 5000, 10000, 20000, 50000]
    
    # Tworzymy największą próbkę, aby mniejsze były jej podzbiorami (nested)
    max_n = max(N_grid)
    X_sample_max = X.sample(n=max_n, random_state=42).copy()
    
    rankings = {}
    importances = {}
    
    for n in N_grid:
        print(f"\nObliczanie wartości SHAP dla N={n}...")
        # Wycinamy prefiks z największej próbki
        X_sample = X_sample_max.iloc[:n]
        
        shap_values = explainer.shap_values(X_sample)
        
        # Obliczenie mean(|SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Zapiszmy importances pod przypisanymi indexami (cechami)
        feature_importance = pd.Series(mean_abs_shap, index=features)
        
        # Sortowanie cech po ważności (malejąco)
        sorted_features = feature_importance.sort_values(ascending=False).index.tolist()
        
        rankings[n] = sorted_features
        importances[n] = feature_importance
        print(f"Top 5 dla N={n}: {sorted_features[:5]}")

    # ==========================================
    # FAZA A - Analiza stabilności (wybór N)
    # ==========================================
    print("\n--- FAZA A: Analiza stabilności ---")
    stability_results = []
    chosen_n = None
    
    for i in range(len(N_grid) - 1):
        n_current = N_grid[i]
        n_next = N_grid[i + 1]
        
        # Wymuszamy tę samą kolejność wektórów, żeby korelacja Spearmana była obliczona poprawnie
        imp_curr = importances[n_current][features].values
        imp_next = importances[n_next][features].values
        
        spearman_corr, _ = spearmanr(imp_curr, imp_next)
        ov10 = overlap_at_k(rankings[n_current], rankings[n_next], 10)
        ov15 = overlap_at_k(rankings[n_current], rankings[n_next], 15)
        
        stability_results.append({
            'N': n_current,
            'spearman_vs_next': spearman_corr,
            'overlap10_vs_next': ov10,
            'overlap15_vs_next': ov15
        })
        
        print(f"N={n_current} vs N={n_next} | Spearman: {spearman_corr:.4f} | Ov@10: {ov10}/10 | Ov@15: {ov15}/15")
        
        # Kryterium wyboru N (pierwszy, który spełni kryterium)
        if chosen_n is None and ov15 == 15 and spearman_corr >= 0.98:
            chosen_n = n_current
            
    # Uzupełniamy tabele w rekord z ostatnim elementem siatki i nan wartościami, jako domknięcie
    stability_results.append({
        'N': N_grid[-1],
        'spearman_vs_next': np.nan,
        'overlap10_vs_next': np.nan,
        'overlap15_vs_next': np.nan
    })
    
    # Zapis wyników
    os.makedirs('artifacts', exist_ok=True)
    stability_df = pd.DataFrame(stability_results)
    stability_df.to_csv('artifacts/shap_stability.csv', index=False)
    print("\nZapisano raport stabilności do artifacts/shap_stability.csv")
    
    if chosen_n is None:
        print("UWAGA: Nie osiągnięto wymaganego kryterium stabilności. Awaryjnie wybierane jest największe N dla Fazy B.")
        chosen_n = N_grid[-1]
    else:
        print(f"Wybrano optymalne N = {chosen_n}")
        
    with open('artifacts/chosen_N.txt', 'w') as f:
        f.write(f"Wybrano N={chosen_n}\n")


    # ==========================================
    # FAZA B - Finalny ranking i sanity check
    # ==========================================
    print(f"\n--- FAZA B: Finałowy ranking i Sanity Check dla wybranego N={chosen_n} ---")
    
    main_top15 = set(rankings[chosen_n][:15])
    sanity_seeds = [43, 44, 45, 46]
    
    for seed in sanity_seeds:
        # Losujemy zupełnie nowy sampel i przepuszczamy przez SHAP
        X_sanity = X.sample(n=chosen_n, random_state=seed)
        sanity_shap = explainer.shap_values(X_sanity)
        sanity_mean_abs = np.abs(sanity_shap).mean(axis=0)
        
        sanity_importance = pd.Series(sanity_mean_abs, index=features).sort_values(ascending=False)
        sanity_top15 = set(sanity_importance.index.tolist()[:15])
        
        ov = len(main_top15 & sanity_top15)
        print(f"   Seed {seed}: Overlap Top-15 między wybranym samplem a głównym rankingiem = {ov}/15")

    # Finalny zrzut cech
    final_ranking = rankings[chosen_n]
    output_rankings = {
        "top_29": final_ranking,
        "top_15": final_ranking[:15],
        "top_10": final_ranking[:10]
    }
    
    with open('artifacts/feature_rankings.json', 'w') as json_file:
        json.dump(output_rankings, json_file, indent=4)
        
    print("\nGotowe. Finałowe rankingi cech (Top-29, Top-15, Top-10) zostały zapisane w artifacts/feature_rankings.json")


if __name__ == "__main__":
    main()