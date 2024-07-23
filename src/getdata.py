import os
import requests
import kaggle

# Crear la carpeta de datos si no existe
def create_data_folders(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for source in ['OpenNeuro', 'Mendeley', 'Kaggle']:
        source_path = os.path.join(base_path, source)
        if not os.path.exists(source_path):
            os.makedirs(source_path)
    print(f"Directorios creados en {base_path}")

# Descargar datos desde OpenNeuro
def download_openneuro_data(dataset_id, save_path):
    base_url = f"https://openneuro.org/crn/datasets/{dataset_id}/snapshots/1.0.0/download"
    response = requests.get(base_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Datos de OpenNeuro descargados y guardados en {save_path}")
    else:
        print(f"Error al descargar datos de OpenNeuro: {response.status_code}")

# Descargar datos desde Mendeley Data
def download_mendeley_data(dataset_id, save_path):
    base_url = f"https://data.mendeley.com/datasets/{dataset_id}/1/download"
    response = requests.get(base_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Datos de Mendeley descargados y guardados en {save_path}")
    else:
        print(f"Error al descargar datos de Mendeley: {response.status_code}")

# Descargar datos desde Kaggle
def download_kaggle_data(dataset, save_path):
    os.environ['KAGGLE_USERNAME'] = 'samuelplatam'
    os.environ['KAGGLE_KEY'] = '416809e1ecd959d48c56f5421c878911'
    kaggle.api.dataset_download_files(dataset, path=save_path, unzip=True)
    print(f"Datos de Kaggle descargados y guardados en {save_path}")

if __name__ == "__main__":
    base_path = "./data"
    create_data_folders(base_path)
    
    # Datos de OpenNeuro
    openneuro_dataset_id = "ds003645"  # Reemplazar con el ID del dataset que deseas descargar
    openneuro_save_path = os.path.join(base_path, 'OpenNeuro', 'eeg_data.zip')
    download_openneuro_data(openneuro_dataset_id, openneuro_save_path)
    
    # Datos de Mendeley Data
    mendeley_datasets = {
        "5pc2j46cbc": "Epileptic_EEG.zip",
        "6k4g25fhzg": "ADHD_EEG.zip",
        "crhybxpdy6": "EEG_fMRI.zip",
        "wnshbvdxs2": "Mental_Stress_EEG.zip"
    }
    for dataset_id, filename in mendeley_datasets.items():
        mendeley_save_path = os.path.join(base_path, 'Mendeley', filename)
        download_mendeley_data(dataset_id, mendeley_save_path)
    
    # Datos de Kaggle (Epilepsy Detection using EEG Signals)
    kaggle_dataset = "oussamabenhassine/epilepsy-detection-using-eeg-signals"
    kaggle_save_path = os.path.join(base_path, 'Kaggle')
    download_kaggle_data(kaggle_dataset, kaggle_save_path)

