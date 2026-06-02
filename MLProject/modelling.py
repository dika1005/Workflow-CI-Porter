import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

def main():
    # 1. Pastikan tracking mengarah ke direktori lokal runner secara eksplisit
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
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
    
    # Matikan autolog total agar tidak terjadi tumpang-tindih context
    mlflow.autolog(disable=True)
    
    print("Memulai pelatihan base model di GitHub Actions...")
    
    # Mengambil Run ID langsung dari Environment Variable bawaan mlflow run
    run_id = os.environ.get("MLFLOW_RUN_ID")
    
    if not run_id:
        print("Menjalankan skrip secara mandiri, membuat Run ID baru...")
        active_run = mlflow.start_run()
        run_id = active_run.info.run_id
    else:
        print(f"Menggunakan Run ID dari Environment System: {run_id}")
    
    # Proses Training
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi metrik
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluasi Base Model ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("----------------------------")
    
    # 3. Mendaftarkan kembali konteks run ke fluently API menggunakan ID yang sama 
    # agar log_model merekam file 'MLmodel' ke meta-database lokal dengan sah.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        print("Mendaftarkan artefak model ke meta-database MLflow...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        
    print(f"Pelatihan selesai. Seluruh komponen Docker siap diekstraksi dari run: {run_id}")

if __name__ == "__main__":
    main()