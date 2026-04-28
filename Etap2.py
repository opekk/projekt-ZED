import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Używamy Keras z TensorFlow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim):
    """
    Buduje model autoenkodera zgodnie z architekturą opisaną w dokumentacji.
    """
    # --- Enkoder ---
    input_layer = Input(shape=(input_dim,), name="input")
    encoded = Dense(100, activation='relu', name="encoded_1")(input_layer)
    encoded = Dense(50, activation='relu', name="encoded_2")(encoded)
    
    # --- Dekoder ---
    decoded = Dense(100, activation='tanh', name="decoded_1")(encoded)
    decoded = Dense(input_dim, activation='relu', name="output")(decoded)
    
    # Kompilacja modelu
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    
    return autoencoder

def main():
    """
    Główna funkcja realizująca Etap 2:
    1. Wczytuje dane i rankingi cech.
    2. Dla każdego zestawu cech (top-29, top-15, top-10):
        a. Przygotowuje i skaluje dane.
        b. Trenuje model autoenkodera.
        c. Oblicza błąd rekonstrukcji dla wszystkich danych.
        d. Generuje etykiety dla progów P={500, 1000, 1500}.
        e. Zapisuje 9 wynikowych zbiorów danych.
    """
    print("--- ETAP 2: Generowanie etykiet za pomocą Autoenkodera ---")

    # 1. Wczytywanie danych
    print("Wczytywanie danych...")
    df = pd.read_csv('creditcard.csv')
    
    # Wczytanie rankingów cech z Etapu 1
    with open('artifacts/feature_rankings.json', 'r') as f:
        feature_rankings = json.load(f)
        
    # Definicja progów P
    P_thresholds = [500, 1000, 1500]
    
    # Przygotowanie katalogu na wyniki
    output_dir = 'artifacts/labeled_datasets'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Wyniki będą zapisywane w: {output_dir}")

    # 2. Pętla po zestawach cech
    for feature_set_name, features in feature_rankings.items():
        print(f"\n--- Przetwarzanie zestawu cech: {feature_set_name} ({len(features)} cech) ---")
        
        # Przygotowanie danych dla bieżącego zestawu cech
        X = df[features]
        
        # Skalowanie danych - kluczowy krok dla sieci neuronowych
        # Cechy V1-V28 są już wyskalowane, ale 'Amount' nie.
        # StandardScaler dopasowujemy do całego zbioru, co jest akceptowalne w podejściu unsupervised.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Podział danych na zbiór treningowy i walidacyjny (zgodnie z dokumentacją)
        # Używamy tylko danych bez oszustw do treningu, aby model nauczył się "normalności"
        # W naszym podejściu unsupervised nie mamy etykiet, więc trenujemy na wszystkich danych.
        # Podział na train/val jest potrzebny do monitorowania treningu.
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        print(f"Kształt danych treningowych: {X_train.shape}")
        print(f"Kształt danych walidacyjnych: {X_val.shape}")

        # 3. Budowa i trening autoenkodera
        input_dim = X_scaled.shape[1]
        autoencoder = build_autoencoder(input_dim)
        
        # Definicja EarlyStopping (zgodnie z dokumentacją)
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=25, 
            restore_best_weights=True,
            verbose=1
        )
        
        print("Rozpoczynanie treningu autoenkodera...")
        history = autoencoder.fit(
            X_train, X_train,
            epochs=250,
            batch_size=256,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 4. Obliczenie błędu rekonstrukcji
        print("Obliczanie błędu rekonstrukcji dla całego zbioru danych...")
        X_reconstructed = autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
        
        # Tworzenie DataFrame z wynikami
        reconstruction_error_df = pd.DataFrame({'reconstruction_error': mse})
        
        # 5. Generowanie etykiet i zapis wyników
        for p in P_thresholds:
            print(f"Generowanie etykiet dla P={p}...")
            
            # Sortowanie po błędzie i wyznaczenie progu
            threshold = reconstruction_error_df['reconstruction_error'].nlargest(p).iloc[-1]
            
            # Stworzenie nowej kolumny z wygenerowanymi etykietami
            generated_labels = (reconstruction_error_df['reconstruction_error'] >= threshold).astype(int)
            
            # Przygotowanie finalnego DataFrame do zapisu
            df_labeled = df.copy()
            df_labeled['generated_class'] = generated_labels
            
            # Sprawdzenie, czy liczba wygenerowanych fraudów jest poprawna
            num_generated_frauds = df_labeled['generated_class'].sum()
            print(f"   Liczba wygenerowanych fraudów: {num_generated_frauds} (oczekiwano ok. {p})")
            
            # Zapis do pliku
            output_filename = f"labeled_data_{feature_set_name}_p{p}.csv"
            output_path = os.path.join(output_dir, output_filename)
            df_labeled.to_csv(output_path, index=False)
            print(f"   Zapisano do pliku: {output_path}")

    print("\n--- Etap 2 zakończony pomyślnie! ---")
    print(f"Wygenerowano i zapisano {len(feature_rankings) * len(P_thresholds)} zbiorów danych w katalogu '{output_dir}'.")


if __name__ == "__main__":
    main()