import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from mpl_toolkits.mplot3d import Axes3D

# Ruta de los datos descargados
data_dir = './data/Kaggle'

# Función para encontrar el archivo CSV
def find_csv_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)
    return None

# Encontrar el archivo CSV
eeg_data_path = find_csv_file(data_dir)
if eeg_data_path is None:
    raise FileNotFoundError("No se encontró ningún archivo CSV en el directorio especificado")

print(f"Archivo encontrado: {eeg_data_path}")

# Cargar datos de EEG
eeg_data = pd.read_csv(eeg_data_path)

# Muestras de datos
print(eeg_data.head())

# Graficar algunas señales EEG
def plot_eeg_signals(data, n_signals=5):
    plt.figure(figsize=(15, 10))
    for i in range(n_signals):
        plt.subplot(n_signals, 1, i+1)
        plt.plot(data.iloc[:, i])
        plt.title(f'Señal EEG {i+1}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
    plt.tight_layout()
    plt.show()

plot_eeg_signals(eeg_data)

# Análisis de frecuencia utilizando la transformada de Fourier
def plot_frequency_analysis(data, sample_rate=256):
    plt.figure(figsize=(15, 10))
    for i in range(5):
        f, Pxx = welch(data.iloc[:, i], fs=sample_rate, nperseg=1024)
        plt.semilogy(f, Pxx, label=f'Señal EEG {i+1}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad espectral de potencia (PSD)')
    plt.legend()
    plt.title('Análisis de frecuencia de señales EEG')
    plt.show()

plot_frequency_analysis(eeg_data)

# Mapas de calor de correlación
def plot_correlation_heatmap(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor de correlación entre canales EEG')
    plt.show()

    # Gráfica en 3D
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(range(corr_matrix.shape[0]), range(corr_matrix.shape[1]))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.5
    dz = corr_matrix.values.flatten()
    
    cmap = plt.get_cmap('coolwarm')
    max_height = np.max(dz)   # obtener el máximo valor de z
    min_height = np.min(dz)   # obtener el mínimo valor de z
    # Escalar alturas
    dz_scaled = (dz - min_height) / (max_height - min_height)
    colors = cmap(dz_scaled)
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
    ax.set_xlabel('Canales EEG')
    ax.set_ylabel('Canales EEG')
    ax.set_zlabel('Correlación')
    ax.set_title('Mapa de calor de correlación en 3D entre canales EEG')
    plt.show()

plot_correlation_heatmap(eeg_data)

# Características del EEG
def extract_eeg_features(data):
    features = pd.DataFrame()
    features['mean'] = data.mean(axis=1)
    features['std'] = data.std(axis=1)
    features['max'] = data.max(axis=1)
    features['min'] = data.min(axis=1)
    features['energy'] = np.sum(data**2, axis=1)
    return features

eeg_features = extract_eeg_features(eeg_data)
print(eeg_features.head())

# Visualización de las características del EEG
def plot_eeg_features(features):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(features.columns):
        plt.subplot(3, 2, i+1)
        plt.hist(features[column], bins=50)
        plt.title(f'Distribución de {column}')
    plt.tight_layout()
    plt.show()

plot_eeg_features(eeg_features)

# Guardar las figuras generadas
def save_figures():
    if not os.path.exists('./neuroscience/figures'):
        os.makedirs('./neuroscience/figures')
    
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(eeg_data.iloc[:, i])
        plt.title(f'Señal EEG {i+1}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
    plt.tight_layout()
    plt.savefig('./neuroscience/figures/eeg_signals.png')
    plt.close()

    plt.figure(figsize=(15, 10))
    for i in range(5):
        f, Pxx = welch(eeg_data.iloc[:, i], fs=256, nperseg=1024)
        plt.semilogy(f, Pxx, label=f'Señal EEG {i+1}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad espectral de potencia (PSD)')
    plt.legend()
    plt.title('Análisis de frecuencia de señales EEG')
    plt.savefig('./neuroscience/figures/frequency_analysis.png')
    plt.close()

    corr_matrix = eeg_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor de correlación entre canales EEG')
    plt.savefig('./neuroscience/figures/correlation_heatmap.png')
    plt.close()

    # Gráfica en 3D
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(range(corr_matrix.shape[0]), range(corr_matrix.shape[1]))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.5
    dz = corr_matrix.values.flatten()
    
    cmap = plt.get_cmap('coolwarm')
    max_height = np.max(dz)   # obtener el máximo valor de z
    min_height = np.min(dz)   # obtener el mínimo valor de z
    # Escalar alturas
    dz_scaled = (dz - min_height) / (max_height - min_height)
    colors = cmap(dz_scaled)
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
    ax.set_xlabel('Canales EEG')
    ax.set_ylabel('Canales EEG')
    ax.set_zlabel('Correlación')
    ax.set_title('Mapa de calor de correlación en 3D entre canales EEG')
    plt.savefig('./neuroscience/figures/correlation_heatmap_3d.png')
    plt.close()

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(eeg_features.columns):
        plt.subplot(3, 2, i+1)
        plt.hist(eeg_features[column], bins=50)
        plt.title(f'Distribución de {column}')
    plt.tight_layout()
    plt.savefig('./neuroscience/figures/eeg_features.png')
    plt.close()

save_figures()
