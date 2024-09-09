# analysis.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy import stats
from scipy.signal import find_peaks
import traceback
from sklearn.metrics import mutual_info_score
from config import (
    VISUALIZATIONS_DIR,
    HEATMAP_CMAP,
    STATISTICAL_RESULTS_DIR,
)

def check_timestamp_issues(data):
    # Check for NaT values in DateTime column
    if data['DateTime'].isnull().any():
        logging.warning("Found missing values in DateTime column. Dropping rows with missing timestamps.")
        data = data.dropna(subset=['DateTime'])
    
    # Check if timestamps are in chronological order
    if not data['DateTime'].is_monotonic_increasing:
        logging.info("Timestamps are not in strict chronological order. This may be normal for your data.")
    
    # Check for duplicate timestamps by DateTime and Hemisphere
    duplicates = data[data.duplicated(subset=['DateTime', 'Hemisphere'], keep=False)]
    if not duplicates.empty:
        logging.info(f"Found {len(duplicates)} potentially duplicate timestamps. This may be normal for your data.")
        # Instead of modifying the data, we'll just log the information
        logging.info("Sample of potential duplicates:")
        logging.info(duplicates.head().to_string())
    
    # Check for large time gaps in the data
    time_diffs = data['DateTime'].diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(days=1)]
    if not large_gaps.empty:
        logging.info(f"Found {len(large_gaps)} large time gaps (>1 day). This may be normal for your data.")
        logging.info("Sample of large gaps:")
        logging.info(large_gaps.head().to_string())

    return data  # Return the data unchanged

def perform_statistical_analysis(data):
    results = {}
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for band in bands:
        left_data = data[data['Hemisphere'] == 'HemisphereLocationDef.Left'][band].dropna()
        right_data = data[data['Hemisphere'] == 'HemisphereLocationDef.Right'][band].dropna()
        if len(left_data) < 2 or len(right_data) < 2:
            logging.warning(f"Insufficient data for t-test in {band} band.")
            results[band] = {'t_statistic': float('nan'), 'p_value': float('nan')}
            continue
        if left_data.var() == 0 or right_data.var() == 0:
            logging.warning(f"No variability in {band} band data.")
            results[band] = {'t_statistic': float('nan'), 'p_value': float('nan')}
            continue
        try:
            t_stat, p_value = stats.ttest_ind(left_data, right_data, nan_policy='omit')
            results[band] = {'t_statistic': t_stat, 'p_value': p_value}
        except Exception as e:
            logging.error(f"Error during t-test for {band} band: {e}")
            results[band] = {'t_statistic': float('nan'), 'p_value': float('nan')}
    return results

def analyze_lfp_frequency_correlation(snapshot_data, lfp_trend_data):
    logging.info("Analyzing LFP and frequency band correlation")
    merged_data = pd.merge_asof(
        lfp_trend_data.sort_values('DateTime'),
        snapshot_data.sort_values('DateTime'),
        on='DateTime',
        by='Hemisphere',
        direction='nearest'
    )
    lfp_column = 'LFP' if 'LFP' in merged_data.columns else ('LFP_x' if 'LFP_x' in merged_data.columns else None)
    if lfp_column is None:
        logging.error("LFP column not found in merged data")
        return None
    required_columns = [lfp_column, 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    missing_columns = [col for col in required_columns if col not in merged_data.columns]
    if missing_columns:
        logging.error(f"Missing columns in merged data: {missing_columns}")
        return None

    correlation_matrix = merged_data[required_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap=HEATMAP_CMAP, vmin=-1, vmax=1, center=0)
    plt.title('Correlation between LFP and Frequency Bands')
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATIONS_DIR}lfp_frequency_correlation_heatmap.png')
    plt.close()
    return correlation_matrix


def generate_strongest_band_table(df, group_by=['PatientID', 'EventName']):
    """
    Generates a table showing the strongest and second strongest frequency bands for each group.

    Args:
        df (pd.DataFrame): The input DataFrame containing snapshot data.
        group_by (list): List of columns to group by, e.g., ['PatientID', 'EventName'].

    Returns:
        pd.DataFrame: A table showing the strongest and second strongest bands per group.
    """
    def find_top_two_bands(group):
        sorted_bands = group[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].mean().sort_values(ascending=False)
        strongest_band = sorted_bands.index[0]
        second_strongest_band = sorted_bands.index[1] if len(sorted_bands) > 1 else None
        return pd.Series([strongest_band, second_strongest_band])

    top_bands = df.groupby(group_by).apply(find_top_two_bands).reset_index()
    top_bands.columns = group_by + ['StrongestBand', 'SecondStrongestBand']
    return top_bands


def generate_mean_power_table(df, group_by=['PatientID', 'Hemisphere']):
    mean_power = df.groupby(group_by)[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].mean().reset_index()
    return mean_power.set_index(group_by)


def generate_statistical_summary(statistical_results):
    summary = []
    for symptom, results in statistical_results.items():
        for band, stats in results.items():
            if isinstance(stats, dict) and 't_statistic' in stats and 'p_value' in stats:
                summary.append({
                    'Symptom': symptom,
                    'Band': band,
                    't-statistic': stats['t_statistic'],
                    'p-value': stats['p_value']
                })
            else:
                logging.warning(f"Unexpected format in stats for {symptom}, {band}: {stats}")
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f'{STATISTICAL_RESULTS_DIR}statistical_summary.csv', index=False)
        logging.info(f'Statistical summary saved to {STATISTICAL_RESULTS_DIR}statistical_summary.csv')
        return summary_df
    else:
        logging.error("No valid statistical data found for summary.")
        return pd.DataFrame()







# Coupling Analysis
def perform_band_coupling_analysis(data, output_dir):
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    results = {}
    
    logging.info("Starting band coupling analysis")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Input data columns: {data.columns}")
    logging.info(f"Input data types:\n{data.dtypes}")
    
    data_copy = data.copy()
    
    for band in bands:
        if band not in data_copy.columns:
            logging.error(f"Column {band} not found in the data.")
            return {}
        logging.info(f"Converting {band} to numeric")
        data_copy[band] = pd.to_numeric(data_copy[band], errors='coerce')
    
    logging.info("Dropping rows with NaN values")
    data_cleaned = data_copy.dropna(subset=bands)
    
    logging.info(f"Cleaned data shape: {data_cleaned.shape}")
    
    if len(data_cleaned) == 0:
        logging.error("No valid data remaining after cleaning.")
        return {}
    
    try:
        logging.info("Calculating Pearson Correlation")
        pearson_corr = data_cleaned[bands].corr(method='pearson')
        results['pearson_correlation'] = pearson_corr
        
        logging.info("Calculating Spearman Correlation")
        spearman_corr = data_cleaned[bands].corr(method='spearman')
        results['spearman_correlation'] = spearman_corr
        
        logging.info("Calculating Power Ratios")
        power_ratios = calculate_power_ratios(data_cleaned[bands])
        results['power_ratios'] = power_ratios
        
        logging.info("Performing Power Peak Analysis")
        power_peaks = perform_power_peak_analysis(data_cleaned[bands])
        results['power_peaks'] = power_peaks
        
        logging.info("Generating visualizations")
        plot_correlation_heatmap(pearson_corr, 'Pearson Correlation', os.path.join(output_dir, 'pearson_correlation_heatmap.png'))
        plot_correlation_heatmap(spearman_corr, 'Spearman Correlation', os.path.join(output_dir, 'spearman_correlation_heatmap.png'))
        plot_power_ratio_heatmap(power_ratios, os.path.join(output_dir, 'power_ratio_heatmap.png'))
        plot_power_peaks(data_cleaned[bands], power_peaks, os.path.join(output_dir, 'power_peaks.png'))
        
        logging.info("Saving results to CSV")
        pearson_corr.to_csv(os.path.join(output_dir, 'pearson_correlation.csv'))
        spearman_corr.to_csv(os.path.join(output_dir, 'spearman_correlation.csv'))
        power_ratios.to_csv(os.path.join(output_dir, 'power_ratios.csv'))
        
        # Save power peaks to CSV
        try:
            power_peaks_df = pd.DataFrame(power_peaks)
            power_peaks_df.to_csv(os.path.join(output_dir, 'power_peaks.csv'), index=False)
        except Exception as e:
            logging.error(f"Error saving power peaks: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")

        
        
    except Exception as e:
        logging.error(f"Error in band coupling analysis: {str(e)}")
        logging.error(f"Error traceback: {traceback.format_exc()}")
        results['error'] = str(e)
    
    return results




def perform_power_peak_analysis(band_data):
    peaks = {}
    for band in band_data.columns:
        peaks[band], _ = find_peaks(band_data[band], distance=20)
    return peaks

def plot_correlation_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_power_ratio_heatmap(power_ratios, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(power_ratios, annot=True, cmap='viridis')
    plt.title('Cross-Frequency Power Ratios')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_power_peaks(band_data, peaks, filename):
    fig, axes = plt.subplots(len(band_data.columns), 1, figsize=(12, 4*len(band_data.columns)), sharex=True)
    for i, band in enumerate(band_data.columns):
        axes[i].plot(band_data[band])
        axes[i].plot(peaks[band], band_data[band].iloc[peaks[band]], "x")
        axes[i].set_title(f'{band} Power Peaks')
        axes[i].set_ylabel('Power')
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    
    
def calculate_power_ratios(band_data):
    logging.info("Calculating power ratios")
    bands = band_data.columns
    ratios = pd.DataFrame(index=bands, columns=bands)
    for band1 in bands:
        for band2 in bands:
            if band1 != band2:
                ratio = (band_data[band1] / band_data[band2]).mean()
                ratios.loc[band1, band2] = ratio if np.isfinite(ratio) else np.nan
    
    logging.info(f"Power ratios data types:\n{ratios.dtypes}")
    logging.info(f"Power ratios shape: {ratios.shape}")
    return ratios



def perform_power_peak_analysis(band_data):
    """Perform power peak analysis for each frequency band."""
    peaks = {}
    for band in band_data.columns:
        peaks[band], _ = find_peaks(band_data[band], distance=20)  # Adjust distance as needed
    return peaks

def plot_correlation_heatmap(corr_matrix, title, filename):
    logging.info(f"Plotting correlation heatmap: {title}")
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        logging.error(f"Error in plot_correlation_heatmap: {str(e)}")
        logging.error(f"Error traceback: {traceback.format_exc()}")
        

def plot_power_ratio_heatmap(power_ratios, filename):
    logging.info("Plotting power ratio heatmap")
    logging.info(f"Power ratios data types before plotting:\n{power_ratios.dtypes}")
    logging.info(f"Power ratios data:\n{power_ratios}")
    
    try:
        # Convert to numeric, replacing any non-numeric values with NaN
        power_ratios_numeric = power_ratios.apply(pd.to_numeric, errors='coerce')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(power_ratios_numeric, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Cross-Frequency Power Ratios')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Power ratio heatmap saved to {filename}")
    except Exception as e:
        logging.error(f"Error in plot_power_ratio_heatmap: {str(e)}")
        logging.exception("Exception traceback:")


def plot_power_peaks(band_data, peaks, filename):
    """Plot and save power peaks for each frequency band."""
    fig, axes = plt.subplots(len(band_data.columns), 1, figsize=(12, 4 * len(band_data.columns)), sharex=True)
    for i, band in enumerate(band_data.columns):
        valid_peaks = [int(idx) for idx in peaks[band] if not np.isnan(idx)]  # Filter out NaN values
        axes[i].plot(band_data[band])
        axes[i].plot(valid_peaks, band_data[band].iloc[valid_peaks], "x")  # Plot only valid peaks
        axes[i].set_title(f'{band} Power Peaks')
        axes[i].set_ylabel('Power')
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def calculate_lfp_symptom_correlation(lfp_trend_data, snapshot_data, time_window=pd.Timedelta(minutes=5)):
    """
    Calculate the statistical significance of correlation between LFP values and specific symptoms.

    :param lfp_trend_data: DataFrame containing LFP trend data
    :param snapshot_data: DataFrame containing snapshot event data
    :param time_window: Time window to consider for LFP values around each symptom event
    :return: DataFrame with correlation results
    """
    results = []

    # Ensure DateTime columns are in datetime format
    lfp_trend_data['DateTime'] = pd.to_datetime(lfp_trend_data['DateTime'])
    snapshot_data['DateTime'] = pd.to_datetime(snapshot_data['DateTime'])

    for patient_id in snapshot_data['PatientID'].unique():
        patient_lfp_data = lfp_trend_data[lfp_trend_data['PatientID'] == patient_id]
        patient_snapshot_data = snapshot_data[snapshot_data['PatientID'] == patient_id]
        
        for symptom in patient_snapshot_data['EventName'].unique():
            symptom_data = patient_snapshot_data[patient_snapshot_data['EventName'] == symptom]
            
            if len(symptom_data) > 1:  # Ensure we have enough data points
                symptom_lfp_values = []
                for event_time in symptom_data['DateTime']:
                    # Find LFP values within the time window of the symptom event
                    window_lfp = patient_lfp_data[
                        (patient_lfp_data['DateTime'] >= event_time - time_window) &
                        (patient_lfp_data['DateTime'] <= event_time + time_window)
                    ]
                    if not window_lfp.empty:
                        symptom_lfp_values.append(window_lfp['LFP'].mean())
                
                if len(symptom_lfp_values) > 1:
                    # Calculate Pearson Correlation
                    correlation, p_value = stats.pearsonr(symptom_lfp_values, range(len(symptom_lfp_values)))
                    
                    # Calculate Spearman Correlation
                    spearman_corr, spearman_p = stats.spearmanr(symptom_lfp_values, range(len(symptom_lfp_values)))
                    
                    # Optional: Mutual Information - ensure arrays are 1D
                    try:
                        mi_score = mutual_info_score(
                            np.array(symptom_lfp_values).ravel(),  # Use .ravel() to ensure it's 1D
                            np.array(range(len(symptom_lfp_values))).ravel()
                        )
                    except ValueError:
                        mi_score = np.nan  # Handle cases where MI calculation fails
                    
                    # Effect size (Cohen's d)
                    effect_size = (np.mean(symptom_lfp_values) - np.mean(patient_lfp_data['LFP'])) / np.std(patient_lfp_data['LFP'])
                    
                    results.append({
                        'PatientID': patient_id,
                        'Symptom': symptom,
                        'Correlation': correlation,
                        'P-value': p_value,
                        'Spearman_Correlation': spearman_corr,
                        'Spearman_P-value': spearman_p,
                        'Mutual_Information': mi_score,
                        'Effect_Size': effect_size,
                        'Sample_Size': len(symptom_lfp_values)
                    })
                else:
                    logging.warning(f"Not enough LFP data points for Patient {patient_id}, Symptom {symptom}")
            else:
                logging.warning(f"Not enough symptom occurrences for Patient {patient_id}, Symptom {symptom}")

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Add interpretation columns if needed
    results_df['Significance'] = np.where(results_df['P-value'] < 0.05, 'Significant', 'Not Significant')
    results_df['Correlation_Strength'] = pd.cut(
        results_df['Correlation'].abs(),
        bins=[-1, 0.3, 0.5, 0.7, 1],
        labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
    )
    
    return results_df


def analyze_shared_events(correlation_results, shared_events):
    """
    Analyze and compare results for shared events across patients.
    
    :param correlation_results: DataFrame with correlation results
    :param shared_events: List of shared event names
    :return: DataFrame with analysis of shared events
    """
    shared_event_results = correlation_results[correlation_results['Symptom'].isin(shared_events)]
    
    analysis_results = []
    for event in shared_events:
        event_data = shared_event_results[shared_event_results['Symptom'] == event]
        if len(event_data) > 1:
            correlation_diff = event_data['Correlation'].diff().iloc[-1]
            p_value_ratio = event_data['P-value'].iloc[0] / event_data['P-value'].iloc[1]
            
            analysis_results.append({
                'Shared_Event': event,
                'Correlation_Difference': correlation_diff,
                'P-value_Ratio': p_value_ratio,
                'Consistent_Significance': event_data['Significance'].nunique() == 1,
                'Consistent_Strength': event_data['Correlation_Strength'].nunique() == 1
            })
    
    return pd.DataFrame(analysis_results)

def interpret_results(correlation_results, shared_events_analysis):
    """
    Interpret the results of the correlation analysis and shared events analysis.
    
    :param correlation_results: DataFrame with correlation results
    :param shared_events_analysis: DataFrame with shared events analysis
    :return: String with interpretation
    """
    interpretation = "LFP-Symptom Correlation Analysis Interpretation:\n\n"
    
    # Overall significant correlations
    significant_correlations = correlation_results[correlation_results['Significance'] == 'Significant']
    interpretation += f"Found {len(significant_correlations)} significant correlations out of {len(correlation_results)} total.\n\n"
    
    # Top correlations
    top_correlations = correlation_results.nlargest(5, 'Correlation')
    interpretation += "Top 5 strongest correlations:\n"
    for _, row in top_correlations.iterrows():
        interpretation += f"- Patient {row['PatientID']}, {row['Symptom']}: r = {row['Correlation']:.2f}, p = {row['P-value']:.4f}\n"
    
    interpretation += "\nShared Events Analysis:\n"
    for _, row in shared_events_analysis.iterrows():
        interpretation += f"- {row['Shared_Event']}:\n"
        interpretation += f"  Correlation Difference: {row['Correlation_Difference']:.2f}\n"
        interpretation += f"  Consistent Significance: {'Yes' if row['Consistent_Significance'] else 'No'}\n"
        interpretation += f"  Consistent Strength: {'Yes' if row['Consistent_Strength'] else 'No'}\n"
    
    return interpretation

def perform_power_peak_analysis(band_data):
    """Perform power peak analysis for each frequency band."""
    peaks = {}
    max_length = 0  # Track the maximum length of peak arrays

    # Find peaks and keep track of the maximum array length
    for band in band_data.columns:
        peak_indices, _ = find_peaks(band_data[band], distance=20)  # Adjust distance as needed
        peaks[band] = peak_indices
        max_length = max(max_length, len(peak_indices))

    # Ensure all arrays have the same length by padding with NaNs
    for band in peaks:
        if len(peaks[band]) < max_length:
            peaks[band] = list(peaks[band]) + [np.nan] * (max_length - len(peaks[band]))

    return peaks
