# Etap 2 pipeline'u: Autoencoder-based label generation

## Context

Po zakończeniu Etapu 1, w którym wygenerowaliśmy rankingi ważności cech (`feature_rankings.json`), przechodzimy do Etapu 2. Jego celem jest wygenerowanie etykiet dla zbioru danych `creditcard.csv` w sposób nienadzorowany.

Etap 2 polega na:

1. Wytrenowaniu trzech osobnych modeli autoenkodera, po jednym dla każdego zestawu cech z Etapu 1 (`top-29`, `top-15`, `top-10`).
2. Użyciu błędu rekonstrukcji (MSE) z każdego autoenkodera jako wskaźnika anomalii (anomaly score) dla wszystkich 290k transakcji.
3. Oznaczeniu `P` transakcji z najwyższym błędem jako "oszustwa" (`Class = 1`), gdzie `P` przyjmuje wartości `{500, 1000, 1500}`.

W rezultacie otrzymamy **9 unikalnych, etykietowanych zbiorów danych** (3 zestawy cech × 3 wartości P), które posłużą jako dane wejściowe do Etapu 3 (ewaluacja jakości etykiet).

---

## Plan implementacji

### 1. Środowisko

Zainstaluj brakujące zależności do `.venv/`. `scikit-learn` i `pandas` powinny być już zainstalowane.

### 2. Struktura kodu

Aby zachować porządek, całą logikę Etapu 2 umieścimy w nowym pliku: `etap2_autoencoder.py`.

#### Kroki implementacyjne w `etap2_autoencoder.py`

##### Wczytanie danych i artefaktów

- Wczytać `creditcard.csv` do `pandas.DataFrame`.
- Wczytać rankingi cech z `feature_rankings.json`.

##### Definicja modelu Autoenkodera

Stworzyć funkcję `build_autoencoder(input_dim)`, która buduje i kompiluje model Keras zgodnie z architekturą z `CLAUDE.md`:

- **Enkoder:** `Input(input_dim)` → `Dense(100, activation='relu')` → `Dense(50, activation='relu')`
- **Dekoder:** `Dense(100, activation='tanh')` → `Dense(input_dim, activation='relu')`
- **Kompilacja:** `optimizer=Adam(learning_rate=1e-4)`, `loss='mean_squared_error'`

##### Główna pętla przetwarzania

Stworzyć pętlę iterującą po trzech zestawach cech (`top_29`, `top_15`, `top_10`). Wewnątrz pętli dla każdego zestawu cech:

**Przygotowanie danych:**
- Wyciąć z głównego DataFrame kolumny odpowiadające bieżącemu zestawowi cech.
- Przeskalować dane za pomocą `StandardScaler` ze scikit-learn. Scaler musi być dopasowany (`fit`) i użyty do transformacji (`transform`) na całym zbiorze.
- Podzielić przeskalowane dane na zbiór treningowy (80%) i walidacyjny (20%) za pomocą `train_test_split`.

**Trening modelu:**
- Zbudować autoenkoder, wywołując `build_autoencoder()` z odpowiednim wymiarem wejściowym.
- Zdefiniować `EarlyStopping` (`patience=25`, `monitor='val_loss'`).
- Wytrenować model (`.fit()`) na danych treningowych, używając danych walidacyjnych do wczesnego zatrzymania (max 250 epok, `batch=256`).

**Obliczenie błędu:**
- Użyć wytrenowanego modelu do predykcji (`.predict()`) na całym przeskalowanym zbiorze danych.
- Obliczyć błąd rekonstrukcji (MSE) dla każdej pojedynczej transakcji.

**Generowanie i zapis etykiet:**

Stworzyć wewnętrzną pętlę po progach `P = [500, 1000, 1500]`. Dla każdego `P`:
- Znaleźć `P` transakcji z najwyższym błędem MSE.
- Stworzyć nową kolumnę `generated_class` w oryginalnym DataFrame, przypisując `1` dla `P` najlepszych anomalii i `0` dla reszty.
- Zapisać cały DataFrame (z nową kolumną) do pliku CSV w katalogu `artifacts/labeled_datasets/`. Nazwa pliku powinna być informacyjna, np. `labeled_data_top15_p1000.csv`.

---

### 3. Szczegóły techniczne

#### Skalowanie

`StandardScaler` jest kluczowy, ponieważ sieć neuronowa jest wrażliwa na skalę cech, a `Amount` ma inną rozpiętość niż cechy `V1-V28`. Dopasowujemy skaler do całego zbioru danych, co jest dopuszczalne, ponieważ jest to krok przetwarzania wstępnego w metodologii nienadzorowanej.

#### Trening Autoenkodera

Model trenujemy, aby odtwarzał dane wejściowe (`X_train` jako wejście i `X_train` jako cel). Zbiór walidacyjny (`X_val`) służy tylko do monitorowania `val_loss` i zatrzymania treningu, gdy model przestaje się poprawiać, co zapobiega przeuczeniu.

#### Generowanie etykiet

Najprostszy sposób to stworzyć `pd.Series` z błędami, posortować, wziąć indeksy `P` największych wartości i na tej podstawie przypisać `1` w nowej kolumnie.

#### Artefakty

- Katalog `artifacts/labeled_datasets/` zawierający 9 plików `.csv`.
- Każdy plik CSV to pełny zbiór `creditcard.csv` plus dodatkowa kolumna `generated_class`.
- Katalog `artifacts` powinien już być w `.gitignore`.

---

### 4. Weryfikacja

| Kryterium | Opis |
|---|---|
| **Liczba plików** | Po uruchomieniu skryptu w katalogu `artifacts/labeled_datasets/` muszą znajdować się **9 plików CSV**. |
| **Poprawność etykiet** | Otwórz losowy plik wynikowy (np. `labeled_data_top15_p1000.csv`) i sprawdź, czy suma wartości w kolumnie `generated_class` jest równa (lub bardzo bliska) `1000`. |
| **Struktura plików** | Sprawdź, czy pliki wynikowe zawierają wszystkie oryginalne kolumny plus nową kolumnę `generated_class`. |
| **Czas wykonania** | Trening trzech autoenkoderów może zająć kilkanaście do kilkudziesięciu minut (w zależności od sprzętu). Użycie GPU znacznie przyspieszy proces. |
