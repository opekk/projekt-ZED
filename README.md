# projekt-ZED

### Introduction

- mamy dataseta zwiazanego z oszustawami bankowymi
- dane w tym datasecie najczęściej są niezbalansowane i nie są poetykietowane. (dataset jest etykietowany, autorzy nie korzystaja z etykiet) 
- naszym celem jest polepszenie etykiet wygenerowanych przez model.
- SHAP zamuje się wymienieniem dancyh najważniejszych dla model.
- Dane w tym datasecie nie mają etykiet, wiec model uczy sie na calym datasecie, korzystając z IS, a potem uzywamy SHAP'a, aby wywnioskowac, które dane mają najważniejsze znaczenie 
- Porównujemy potem wyniki SHAP'a z wynikami IS (Isolation Forest)


--
### Methodology 

- metodologia uzwględnia tylko pracę na niezbalansowanych zbiorach danych
- metoda SHapley Additive exPlanations (SHAP) pozwala nam na zrozumienie pracy modelu, poprzez nadawanie wyniku (shap_value) każdemu z cech. Najważniejsze cechy dostają najwięcej punktów, przez co wiemy, jak dana cecha wpływa na wynik końcowy modelu uczącego. \

--
- najpierw wyuczono model korzystając z Isolation Forest, który bazował na wszystkich cechach zbioru. 
- potem skorzystano z metody SHAP, aby uwzględnić ważność danych cech i jak bardzo one wpływają na model. 
- Korzystamy z abs(shap_value), aby potem uśrednić wynik, dzięki czemu dostajemy wartość ważności cechy. 
- Bierzemy grupy najważniejszych shap_values, i wyuczamy model IS na tych grupach. Grupy to: [3,5,7,10,15,29] cech.
- 50 replik nauki, 300 całościowych outputów
- analiza wariantów poprzez HSD test. 
- Wyniki pokazały, że datasety z 15 i 29 cechami dały statystycznie podobną jakość etykiet, co oznacza, że dodatkowe 14 cech nie wnosi istotnej wartości. Dataset z 10 cechami wypadł nieco gorzej, ale nadal był użyteczny.