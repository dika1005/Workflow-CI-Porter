# Membangun_model/modelling.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

def main():
    # 1. Atur lokasi penyimpanan eksperimen MLflow secara lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Porter_Delivery_Base_Model")

    # 2. Membaca dataset hasil preprocessing
    data_path = "namadataset_preprocessing/porter_delivery_preprocessed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Split Fitur dan Target (Target variabel kita adalah delivery_time_minutes)
    X = df.drop(columns=['delivery_time_minutes'])
    y = df['delivery_time_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Aktifkan MLflow Autolog sebelum inisialisasi model
    mlflow.autolog()
    
    # 4. Melatih Base Model menggunakan Random Forest Regressor dengan parameter standar
    print("Memulai pelatihan base model dengan MLflow Autolog...")
    with mlflow.start_run(run_name="RandomForest_Base"):
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi sederhana
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n--- Evaluasi Base Model ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print("----------------------------")
        print("Pelatihan selesai. Semua metrik dan parameter dicatat otomatis oleh Autolog.")

if __name__ == "__main__":
    main()