# projekt-ZED

### Notatki dotyczące papieru

### Introduction

- mamy dataseta zwiazanego z oszustawami bankowymi
- dane w tym datasecie najczęściej są niezbalansowane i nie są poetykietowane. (dataset jest etykietowany, autorzy nie korzystaja z etykiet) 
- naszym celem jest polepszenie etykiet wygenerowanych przez model.
- SHAP zamuje się wymienieniem dancyh najważniejszych dla model.
- Dane w tym datasecie nie mają etykiet, wiec model uczy sie na calym datasecie, korzystając z IS, a potem uzywamy SHAP'a, aby wywnioskowac, które dane mają najważniejsze znaczenie 
- Porównujemy potem wyniki SHAP'a z wynikami IS (Isolation Forest)

###