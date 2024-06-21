import os
import pandas as pd

BASE_PATH = 'C:\\Users\\ASUS\\Desktop\\pytorchproject\\pythonProject1'
DISTANZE_PATH = os.path.join(BASE_PATH, 'matrice_distanze_magazzino.xlsx')
PALLET_PATH = os.path.join(BASE_PATH, 'Cartel2.xlsx')
LOG_FILE = 'magazzino.log'

def load_pallet_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading pallet data from {file_path}: {e}")

def load_distanze(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        xls = pd.ExcelFile(file_path)
        distanze = pd.read_excel(xls, xls.sheet_names[0])
        return distanze
    except Exception as e:
        raise RuntimeError(f"Error loading distanze data from {file_path}: {e}")

# Caricamento dei dati all'avvio
pallet_data = load_pallet_data(PALLET_PATH)
distanze = load_distanze(DISTANZE_PATH)
