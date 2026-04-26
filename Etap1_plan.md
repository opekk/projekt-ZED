# Etap 1 pipeline'u: Isolation Forest + SHAP feature selection

## Context

Projekt reprodukuje pipeline z artykułu `s40537-025-01154-1.pdf`: label refinement dla niezbalansowanego zbioru Kaggle Credit Card Fraud (`creditcard.csv`, ~290k wierszy, 30 cech). Ground-truth etykiety są dostępne, ale **nie mogą być używane na etapie treningu** — służą tylko do końcowej ewaluacji jakości wygenerowanych etykiet.

Etap 1 pipeline'u polega na:
1. Wytrenowaniu Isolation Forest na 29 cechach (`Amount`, `V1`–`V28`).
2. Obliczeniu `mean(|SHAP|)` per cecha i wyprodukowaniu trzech rankingów: top-29 (all), top-15, top-10. Te rankingi są wejściem do etapu 2 (autoenkoder).

Oryginalny problem, który trzeba było rozwiązać: uruchomienie SHAP na pełnym 290k było uważane za zbyt kosztowne. Rozważano stratyfikowany sample `wszystkie fraudy + 10k non-fraud`, ale to stanowi wyciek etykiet (sample selection bias) — używanie kolumny `Class` do konstrukcji trainingu łamie założenie unsupervised methodology paperu, nawet jeśli kolumna zostanie dropnięta przed fit. **Decyzja:** losowy sample bez stratyfikacji — czysto unsupervised. Zamiast arbitralnie wybierać rozmiar sample'a (np. 20k), **rozmiar N dobieramy empirycznie przez analizę stabilności rankingu** — liczymy krzywą zbieżności top-15 i wybieramy N, przy którym ranking wchodzi w plateau. Daje to obronialną metodologicznie wartość N (można ją zilustrować wykresem w pracy), zamiast zgadywanki.

## Plan implementacji

### 1. Środowisko
Zainstaluj brakujące zależności do `.venv/`:
```bash
.venv/bin/pip install scikit-learn shap
```

### 2. Struktura kodu
Na tym etapie trzymaj się minimalnej struktury — wszystko w `main.py`, bez przedwczesnego rozbijania na moduły. Dopiero gdy etap 2 (autoenkoder) zacznie być dokładany, wydziel moduły.

Etap 1 dzieli się na dwie fazy: **(A) analiza stabilności → wybór N**, potem **(B) finalny ranking na wybranym N**.

#### Faza A — analiza stabilności (wybór N)

- Wczytać `creditcard.csv` raz (~150 MB — unikaj wielokrotnego I/O).
- Wytrenować IF **raz** na pełnym 290k z `Amount` + `V1`–`V28` (bez `Time`, bez `Class`). Ten sam model będzie używany w całej analizie — samplujemy tylko dane wejściowe do SHAP.
- Zdefiniować siatkę rozmiarów sample'a: `N_grid = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]`.
- **Sample nested** (kluczowe dla redukcji szumu porównania): losuj największy sample raz (50k, seed=42), a mniejsze sample'y twórz jako jego prefiksy po wcześniejszym przetasowaniu — wtedy `N=1k ⊂ 2k ⊂ 5k ⊂ ... ⊂ 50k` i różnice w rankingu wynikają z liczby punktów, nie z losowania innych wierszy.
- Dla każdego N policzyć SHAP values (`shap.TreeExplainer` na tym samym modelu IF) i `mean(|SHAP|)` → ranking cech.
- Zmierzyć stabilność między kolejnymi N:
  - **Spearman correlation** pełnych rankingów (29 cech) — oczekujemy zbiegania do ~1.0.
  - **Overlap@10** i **Overlap@15** (liczba wspólnych cech w top-k) — najistotniejsze metryki, bo to te feature sety idą do etapu 2.
- **Kryterium wyboru N:** najmniejsze N, przy którym **overlap@15 = 15** (identyczny top-15) względem kolejnego, większego N w siatce, oraz Spearman ≥ 0.98. Jeśli żadne N nie spełnia, zwiększ siatkę o `100_000`.
- Zapisać krzywą zbieżności (tabelka `N → spearman, overlap@10, overlap@15`) do `artifacts/shap_stability.csv` — to materiał do wykresu w pracy.

#### Faza B — finalny ranking

- Użyć N wybranego w Fazie A.
- Policzyć SHAP na sample'u tego rozmiaru (ten sam seed i sample co w Fazie A, żeby rezultat był w 100% tym samym rankingiem, który został zweryfikowany jako stabilny — nie liczyć ponownie).
- **Multi-seed sanity check:** dla wybranego N wylosuj dodatkowo 4 sample'y tego samego rozmiaru z różnymi seedami (np. 43, 44, 45, 46), policz SHAP i sprawdź, że top-15 jest identyczny lub różni się maksymalnie o 1–2 cechy na końcu listy. To potwierdza, że ranking nie zależy od konkretnego składu sample'a (w tym od przypadkowej liczby fraudów w sample'u — patrz niżej). Jeśli top-15 skacze, zwiększ N.
- Zrzucić trzy feature listy (top-29, top-15, top-10) do `artifacts/feature_rankings.json`, żeby etap 2 mógł je załadować bez ponownego liczenia SHAP.

#### Uwaga metodologiczna: brak fraudów w sample'u

Sample jest losowy, więc przy małych N może nie zawierać żadnego fraudu (prob. ~18% dla N=1k, ~3% dla N=2k, pomijalne dla N≥10k). **To nie łamie metodologii** z dwóch powodów:

1. **Isolation Forest jest trenowany raz na pełnym 290k** — struktura drzew izolująca fraudy jest już w modelu, niezależnie od tego co pokażemy później SHAP-owi.
2. **SHAP wyjaśnia model, nie etykiety.** `mean(|SHAP|)` mierzy, jak mocno cecha wpływa na anomaly score (wartość absolutna — kierunek nie ma znaczenia). Dla non-fraud points te same cechy, które izolują fraudy (`V14`, `V12`, `V10`), nadal mają wysokie `|SHAP|`, bo ich wartości decydują o ścieżce w drzewie.

Gdyby jednak przy małych N ranking był niestabilny, wykryje to Faza A (spadek `overlap@15`) oraz multi-seed sanity check z Fazy B.

### 3. Szczegóły techniczne

**Isolation Forest:**
- `IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)` — paper nie narzuca szczegółów IF, `contamination='auto'` jest bezpieczne.
- Fit na pełnym `X` (290k, 29 cech) — trwa sekundy. **Fit robimy raz**, niezależnie od fazy.

**SHAP:**
- `explainer = shap.TreeExplainer(isolation_forest_model)` — działa z IF w nowszych wersjach `shap`.
- `shap_values = explainer.shap_values(X_sample)`.
- Ranking: `np.abs(shap_values).mean(axis=0)` → mapowanie na nazwy kolumn → `argsort` malejąco.
- Dla Fazy A przewidywany total koszt SHAP: ~50k + 20k + 10k + 5k + 2k + 1k ≈ 88k eksplanacji łącznie; TreeSHAP na IF liczy to w minutach, nie godzinach.

**Metryki stabilności (pseudokod):**
```python
from scipy.stats import spearmanr

def overlap_at_k(ranking_a, ranking_b, k):
    return len(set(ranking_a[:k]) & set(ranking_b[:k]))

# dla kolejnych par (N_i, N_{i+1}):
spearman = spearmanr(importance_N_i, importance_N_next).correlation
ov10 = overlap_at_k(rank_N_i, rank_N_next, 10)
ov15 = overlap_at_k(rank_N_i, rank_N_next, 15)
```

**Artefakty:**
- `artifacts/feature_rankings.json` — klucze `"top_29"`, `"top_15"`, `"top_10"` z listami nazw cech w kolejności ważności.
- `artifacts/shap_stability.csv` — kolumny `N, spearman_vs_next, overlap10_vs_next, overlap15_vs_next`; input do wykresu w pracy.
- `artifacts/chosen_N.txt` (opcjonalnie) — wybrane N z krótkim uzasadnieniem.
- Katalog `artifacts/` dodaj do `.gitignore` (jak `creditcard.csv`) — to są duże/odtwarzalne wyniki, nie code.

### 4. Krytyczne pliki
- `main.py` — implementacja etapu 1 (obie fazy; można rozbić na funkcje `run_stability_analysis()` i `run_final_ranking()`, ale nadal w jednym pliku).
- `.gitignore` — dopisać `artifacts/` (oraz potwierdzić, że `.venv/` jest tam).
- `creditcard.csv` — tylko read, nie modyfikować.

## Weryfikacja

1. **Sanity check czasowy:** cała Faza A (analiza stabilności, 6 rozmiarów N) powinna domknąć się w < 15 min; Faza B to już tylko zapis do JSON.
2. **Reprodukowalność:** uruchomienie dwukrotnie z tym samym `random_state` musi dać identyczną krzywą stabilności i identyczny finalny ranking.
3. **Krzywa stabilności ma sens:** `spearman` i `overlap@15` rosną monotonicznie z N i osiągają plateau przed końcem siatki. Jeśli `overlap@15` skacze niemonotonicznie lub nigdy nie osiąga 15, coś jest nie tak z sample'owaniem lub SHAP-em.
4. **Sanity check merytoryczny:** top-10 cech powinno zawierać przynajmniej kilka z `V14, V12, V10, V17, V4, V11` — cechy znane z literatury jako najbardziej informatywne dla tego datasetu. Jeśli zupełnie ich nie ma w top-10, prawdopodobnie coś jest nie tak (np. zapomniane skalowanie `Amount` albo błąd w SHAP).
5. **Porównanie z paperem:** jeśli paper publikuje swój top-15, porównać overlap — wysoki overlap uwiarygadnia implementację.
6. **Artefakty:** `artifacts/feature_rankings.json` (trzy listy: 29, 15, 10) oraz `artifacts/shap_stability.csv` (krzywa zbieżności) istnieją.

## Co NIE jest w tym planie (świadomie)

- Etap 2 (autoenkoder, 9 datasetów) i etap 3 (4 klasyfikatory × 50 foldów) — zostaną zaplanowane osobno, gdy etap 1 będzie gotowy i zweryfikowany.
- Testy jednostkowe — na tym etapie nie ma jeszcze sensownej jednostki do testowania; dodamy pytest dopiero przy etapie 2/3, gdzie są pipeline boundaries warte pokrycia.
- Logging/konfiguracja — minimalne `print`'y wystarczą dopóki jest jeden plik.
