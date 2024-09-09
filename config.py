# config.py

# Logging configuration
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOGGING_LEVEL = 'INFO'  # Change to 'DEBUG' for more detailed logs

# Directory paths
DATA_DIR = 'data/'                    # Directory for input data files
OUTPUT_DIR = 'output/'                # Directory for output data
VISUALIZATIONS_DIR = f'{OUTPUT_DIR}visualizations/'  # Directory for saving visualizations
STATISTICAL_RESULTS_DIR = f'{OUTPUT_DIR}statistical_results/'  # Directory for saving statistical results
BAND_COUPLING_DIR = f'{OUTPUT_DIR}band_coupling/'  # Directory for saving band coupling results

# File paths
COMBINED_SNAPSHOT_DATA_CSV = f'{OUTPUT_DIR}combined_snapshot_data.csv'
COMBINED_LFP_TREND_DATA_CSV = f'{OUTPUT_DIR}combined_lfp_trend_data.csv'
STRONGEST_BAND_TABLE_CSV = f'{OUTPUT_DIR}strongest_band_table.csv'
MEAN_POWER_TABLE_CSV = f'{OUTPUT_DIR}mean_power_table.csv'
STATISTICAL_SUMMARY_CSV = f'{OUTPUT_DIR}statistical_summary.csv'

# Frequency bands definitions
FREQUENCY_BANDS = {
    'Delta': (0, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

# Plotting configurations
PLOT_FIGSIZE = (10, 8)  # Default size for plots
HEATMAP_CMAP = 'coolwarm'  # Color map for heatmaps
