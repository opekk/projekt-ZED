# projekt-ZED

### Introduction

- Mamy dataseta zwiazanego z oszustawami kart kredytowych.
- Dane w tym datasecie najczęściej są niezbalansowane i nie są poetykietowane. 
(dataset jest etykietowany, autorzy nie korzystaja z etykiet).
- Naszym celem jest polepszenie etykiet wygenerowanych przez model.
- SHAP zajmuje się wymienieniem danych najważniejszych dla model.
- Dane w tym datasecie nie mają etykiet, wiec model uczy sie na calym datasecie, korzystając z IF, a potem uzywamy SHAP'a, aby wywnioskowac, które dane mają najważniejsze znaczenie.
- Porównujemy potem wyniki SHAP'a z wynikami IF (Isolation Forest).

---

### Methodology

- Metodologia uzwględnia tylko pracę na niezbalansowanych zbiorach danych.
- Metoda SHapley Additive exPlanations (SHAP) pozwala nam na zrozumienie pracy modelu, poprzez nadawanie wyniku (shap_value) każdemu z cech. Najważniejsze cechy dostają najwięcej punktów, przez co wiemy, jak dana cecha wpływa na wynik końcowy modelu uczącego.

#### Isolation Forest + SHAP

- Najpierw wyuczono model korzystając z Isolation Forest, który bazował na wszystkich cechach zbioru.
- Potem skorzystano z metody SHAP, aby uwzględnić ważność danych cech i jak bardzo one wpływają na model.
- Korzystamy z `abs(shap_value)`, aby potem uśrednić wynik, dzięki czemu dostajemy wartość ważności cechy.

- Dane dla poprzedniego badania:
- Bierzemy grupy najważniejszych shap_values, i wyuczamy model IF na tych grupach. Grupy to: [3, 5, 7, 10, 15, 29] cech.
- 50 replik nauki, 300 całościowych outputów.

- Analiza wariantów poprzez HSD test.
- Wyniki pokazały, że datasety z 15 i 29 cechami dały statystycznie podobną jakość etykiet. Dataset z 10 cechami wypadł nieco gorzej, ale nadal był użyteczny. Autorzy wybrali te 3 do dalszej analizy.

#### Autoenkoder

- Autoenkoder uczy się odtwarzać dane wejściowe. Po trenowaniu obliczany jest błąd rekonstrukcji (MSE) dla każdej transakcji. Instancje z wysokim błędem są klasyfikowane jako potencjalne oszustwa, a te z niskim — jako normalne transakcje. Ustalany jest próg, powyżej którego transakcje otrzymują etykietę 'fraud'. Dodatkowo część instancji tuż poniżej progu jest też oznaczana jako fraud, żeby wzbogacić klasę mniejszościową.
- Parametry Autoenkodera:
  - Learning rate: 0.0001
  - Batch size: 256
  - Optymalizator: Adam
  - Funkcja straty: MSE (Mean Squared Error)
  - Maksymalna liczba epok: 250
  - EarlyStopping patience: 25 epok
  - Zbiór walidacyjny: 20% danych
  - Architektura enkodera: wejście (29/15/10) → 100 neuronów (ReLU) → 50 neuronów (ReLU)
  - Architektura dekodera: 50 → 100 (Tanh) → wyjście (ReLU)
  - Framework: Keras 2.8.0
- Wynik: dataset z 29, 15 i 10 cechami, dla kazdego datasetu P = 500, 1000, 1500. Ogólnie 9 datasetów.

---

### Experiment

- Kaggle dataset, około 290k rekordów, anonimowe, podczas procesu nie bierzemy etykiet.
- 30 cech: Amount, Time, V1-V28 — zanonimizowane poprzez algorytm PCA.
- Pod uwage bierzemy ceche Amount i V1-V28.
- Poprzez SHAP kategoryzujemy cechy V* od najwazniejszej do najmniej.
- TOP15: V13, V26, Amount, V22, V2, V24, V18, V19, V12, V3, V25, V27, V10, V16, and V4.

---

### Measuring label quality

- Generujemy etykiety dla różnych wartosci P (P — liczba instacji typu Fraud). Większe P — więcej transakcji oznaczonych Fraud.
- Działamy SHAP'em na datasecie z 10, 15 i 29 cechami, aby porównać wpływ cech na jakość etykiet.
- Na wygenerowanych etykietach trenujemy klasyfikatory, dzięki czemu możemy określić jakość etykiet.
- Wyniki porównujemy z Isolation Forest jako baseline.

---

### Performance Metrics

- Korzystamy z AUPRC (Area Under the Precision-Recall Curve), który jest obliczany na podstawie True Positive, False Positive, False Negative i True Negative z macierzy pomyłek.
- AUPRC to metryka, która podsumowuje kompromis między precyzją (precision) a czułością (recall) modelu — im wyższa wartość AUPRC, tym lepiej model radzi sobie z wykrywaniem oszustw.

---

### Classifiers

- Korzystamy z 4 modeli: Decision Tree, Random Forest, Logistic Regression i Multi-Layer Perceptron, aby określić dokładność wygenerowanych etykiet.
- Porównujemy wyniki z tymi modelami z Isolation Forest, które uczy się nienadzorowanie, tak aby zbadać samą dokładność wygenerowanych etykiet.
- Nie korzystamy ze strojenia hiperparametrów, chcemy sprawdzić ocene etykiet.
- 5-krotna walidacja krzyżowa powtórzona 10 razy w celu sprawdzenia czy wygenerowane etykiety są blisko prawdziwych etykiet.

---

### Experimental setup

- Korzystamy z SHAPA, ktory wytworzyl 3 datasety: 29 cech, 15 cech i 10 cech.
- Korzystamy z uczenia nienadzorowanego do kazdego z wytworzonych datasetów, przez co możemy zmierzyć selekcje wygenerowanych etykiet.
- Następnie bierzemy liczbe transakcji oznaczonych jako Fraud (P) dla kazdego z datasetów. P = 500, 1000, 1500. Plateau przychodzi dla P = 1500.
- Utworzono 9 datasetów, 5 modele uczące, 50 walidacji krzyżowych → 2250 outcomów.
- Walidacja krzyżowa była dzielona na 80/20 → 80% dane uczące, 20% dane testowe.
- Modele nadzorowane trenowane w 80% na wygenerowanych etykietach, 20% z orginalnymi etykietami.