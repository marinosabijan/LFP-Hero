# data_processing.py

import json
import pandas as pd
import logging
from config import DATA_DIR, FREQUENCY_BANDS

def load_json_data(file_path):
    """
    Load data from a JSON file.
    
    :param file_path: Path to the JSON file
    :return: Loaded JSON data
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logging.info(f'Data successfully loaded from {file_path}')
        return data
    except FileNotFoundError:
        logging.error(f'File not found: {file_path}')
        return None
    except json.JSONDecodeError:
        logging.error(f'Error decoding JSON from {file_path}')
        return None

def calculate_band_powers(frequencies, fft_bin_data):
    """
    Calculate power within defined frequency bands from FFT data.
    
    :param frequencies: List of frequencies
    :param fft_bin_data: Corresponding FFT bin data
    :return: Tuple of dictionary with band powers, most powerful band, and its power
    """
    band_powers = {band: 0 for band in FREQUENCY_BANDS}
    for freq, power in zip(frequencies, fft_bin_data):
        for band, (low, high) in FREQUENCY_BANDS.items():
            if low <= freq < high:
                band_powers[band] += power
                break

    max_band = max(band_powers, key=band_powers.get)
    max_power = band_powers[max_band]
    return band_powers, max_band, max_power

def extract_snapshot_data(data, patient_id):
    """
    Extract and process LFP snapshot data from JSON structure.

    :param data: Loaded JSON data
    :param patient_id: Identifier for the patient
    :return: DataFrame with processed snapshot data
    """
    if not data:
        return pd.DataFrame()

    lfp_snapshot_events = data.get("DiagnosticData", {}).get("LfpFrequencySnapshotEvents", [])
    records = []
    for event in lfp_snapshot_events:
        common_details = {
            'DateTime': pd.to_datetime(event.get('DateTime')),
            'PatientID': patient_id,
            'EventID': event.get('EventID'),
            'EventName': event.get('EventName'),
            'Cycling': event.get('Cycling', False),
            'LFP': event.get('LFP', False)
        }
        for hemisphere, details in event.get('LfpFrequencySnapshotEvents', {}).items():
            frequencies = details.get('Frequency', [])
            fft_bin_data = details.get('FFTBinData', [])
            band_powers, max_band, max_power = calculate_band_powers(frequencies, fft_bin_data)
            
            for freq, power in zip(frequencies, fft_bin_data):
                records.append({
                    **common_details,
                    'Hemisphere': hemisphere,
                    'GroupId': details.get('GroupId'),
                    'SenseID': details.get('SenseID'),
                    'Frequency': freq,
                    'Power': power,
                    'MaxBand': max_band,
                    'MaxPower': max_power,
                    **band_powers
                })

    return pd.DataFrame(records)

def extract_lfp_trend_data(data, patient_id):
    """
    Extract LFP trend data from JSON structure.

    :param data: Loaded JSON data
    :param patient_id: Identifier for the patient
    :return: DataFrame with processed LFP trend data
    """
    logging.info(f"Extracting LFP trend data for patient {patient_id}")
    lfp_trend_logs = data.get("DiagnosticData", {}).get("LFPTrendLogs", {})
    records = []
    for hemisphere, logs in lfp_trend_logs.items():
        for date, entries in logs.items():
            for entry in entries:
                records.append({
                    'DateTime': pd.to_datetime(entry['DateTime']),
                    'Hemisphere': hemisphere,
                    'LFP': entry['LFP'],
                    'AmplitudeInMilliAmps': entry['AmplitudeInMilliAmps'],
                    'PatientID': patient_id
                })
    df = pd.DataFrame(records)
    logging.info(f"Extracted LFP trend data shape for patient {patient_id}: {df.shape}")
    return df