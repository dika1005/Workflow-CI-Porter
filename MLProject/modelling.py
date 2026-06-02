import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def main():
    # 1. Mengatur nama eksperimen lokal di runner
    mlflow.set_experiment("Porter_Delivery_Base_Model")

    # 2. Membaca dataset
    data_path = "porter-delivery-time-estimation_preprcessing/porter_delivery_preprocessed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di jalur: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Split Fitur dan Target
    X = df.drop(columns=['delivery_time_minutes'])
    y = df['delivery_time_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Matikan autolog total agar tidak mengintervensi backend
    mlflow.autolog(disable=True)
    
    print("Memulai pelatihan base model di GitHub Actions...")
    
    # Ambil ID run aktif yang sudah disediakan oleh perintah 'mlflow run'
    active_run = mlflow.active_run()
    if not active_run:
        raise RuntimeError("Tidak ada run aktif dari MLflow Project.")
    
    run_id = active_run.info.run_id
    print(f"Menggunakan low-level MlflowClient untuk Run ID: {run_id}")
    
    # Inisialisasi MlflowClient
    client = MlflowClient()
        
    # Proses Training
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi metrik
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 3. Catat parameter dan metrik secara aman menggunakan MlflowClient
    client.log_param(run_id, "n_estimators", 50)
    client.log_param(run_id, "max_depth", 10)
    client.log_metric(run_id, "rmse", rmse)
    client.log_metric(run_id, "mae", mae)
    client.log_metric(run_id, "r2", r2)
    
    print("\n--- Evaluasi Base Model ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("----------------------------")
    
    # 4. Simpan biner model langsung ke dalam direktori artefak lokal milik run_id tersebut
    print("Menyimpan artefak model secara eksplisit...")
    # Menghitung jalur lokal absolut tempat mlflow run menyimpan artefaknya
    local_artifact_dir = f"mlruns/0/{run_id}/artifacts/model"
    mlflow.sklearn.save_model(sk_model=model, path=local_artifact_dir)
    print(f"Pelatihan selesai. Artefak berhasil dikunci secara fisik di: {local_artifact_dir}")

if __name__ == "__main__":
    main()