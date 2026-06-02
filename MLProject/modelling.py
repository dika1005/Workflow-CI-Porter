import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

def main():
    # 1. Aktifkan Autolog di paling awal. 
    # Saat dipanggil lewat 'mlflow run', Autolog otomatis merekam semuanya ke run aktif bawaan sistem!
    mlflow.autolog()

    # 2. Membaca dataset
    data_path = "porter-delivery-time-estimation_preprocessing/porter_delivery_preprocessed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di jalur: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Split Fitur dan Target
    X = df.drop(columns=['delivery_time_minutes'])
    y = df['delivery_time_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Memulai pelatihan base model dengan MLflow Autolog di GitHub Actions...")
    
    # 3. Langsung latih model tanpa dibungkus dengan 'with mlflow.start_run()' manual
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Prediksi untuk memunculkan log output evaluasi di console GitHub Actions
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluasi Base Model ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("----------------------------")
    print("Pelatihan selesai. Autolog berhasil mengunci berkas 'MLmodel' di folder mlruns!")

if __name__ == "__main__":
    main()