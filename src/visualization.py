import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists; if not, create it.
    """
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Directory checked/created: {directory}")

def save_plot(fig, filename):
    """
    Save the plot, overwriting if the file exists.
    """
    if os.path.exists(filename):
        os.remove(filename)  # Remove the file if it exists to ensure a fresh save
        logging.info(f"Overwriting existing plot: {filename}")
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)  # Reduce padding and trim white space
    plt.close(fig)
    logging.info(f"Plot saved: {filename}")

#trend event plot
def plot_lfp_trend_with_events(lfp_trend_data, snapshot_data, patient_id, output_dir):
    """
    Plot LFP trend with mapped event snapshots for a single patient.

    :param lfp_trend_data: DataFrame containing LFP trend data
    :param snapshot_data: DataFrame containing snapshot event data
    :param patient_id: Identifier for the patient
    :param output_dir: Directory to save the output plot
    """
    try:
        logging.info(f"Starting plot_lfp_trend_with_events for Patient {patient_id}")
        
        # Filter data for the specific patient
        lfp_trend_data = lfp_trend_data[lfp_trend_data['PatientID'] == patient_id]
        snapshot_data = snapshot_data[snapshot_data['PatientID'] == patient_id]

        logging.info(f"LFP trend data shape: {lfp_trend_data.shape}")
        logging.info(f"Snapshot data shape: {snapshot_data.shape}")

        # Ensure DateTime columns are datetime type
        lfp_trend_data['DateTime'] = pd.to_datetime(lfp_trend_data['DateTime'])
        snapshot_data['DateTime'] = pd.to_datetime(snapshot_data['DateTime'])

        # Merge LFP values for both hemispheres
        lfp_trend_data = lfp_trend_data.groupby('DateTime')['LFP'].mean().reset_index()

        logging.info("Creating plot")
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot LFP trend
        ax.plot(lfp_trend_data['DateTime'], lfp_trend_data['LFP'], label='LFP Trend', color='blue', alpha=0.7)

        # Map event snapshots
        unique_events = snapshot_data['EventName'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_events)))
        color_dict = dict(zip(unique_events, colors))

        # Define a time window for matching (10 minutes)
        time_window = pd.Timedelta(minutes=10)

        for _, event in snapshot_data.iterrows():
            event_time = event['DateTime']
            event_name = event['EventName']
            
            # Find the closest LFP data point within the time window
            closest_lfp = lfp_trend_data[
                (lfp_trend_data['DateTime'] >= event_time - time_window) &
                (lfp_trend_data['DateTime'] <= event_time + time_window)
            ]
            
            if not closest_lfp.empty:
                closest_time = closest_lfp.iloc[0]['DateTime']
                lfp_value = closest_lfp.iloc[0]['LFP']
                ax.scatter(closest_time, lfp_value, color=color_dict[event_name], s=100, label=event_name, zorder=5)

        logging.info("Customizing plot")
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('LFP Value')
        ax.set_title(f'LFP Trend with Event Snapshots - Patient {patient_id}')
        
        # Handle legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()

        logging.info("Saving plot")
        output_file = os.path.join(output_dir, f'lfp_trend_with_events_patient_{patient_id}.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logging.info(f"LFP trend with events plot saved for Patient {patient_id}")
    
    except Exception as e:
        logging.error(f"Error in plot_lfp_trend_with_events for Patient {patient_id}: {str(e)}")
        logging.error(traceback.format_exc())
        plt.close()  # Make sure to close the plot even if an error occurs



def plot_heatmap(data, title, patient_id, output_dir):
    mean_powers = data.groupby('EventName')[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].mean()

    patient_dir = os.path.join(output_dir, 'output', f'patient_{patient_id}')
    group_dir = os.path.join(output_dir, 'output', 'grouped_patients')

    ensure_directory_exists(patient_dir)
    ensure_directory_exists(group_dir)
    save_dir = patient_dir if os.path.exists(patient_dir) else group_dir

    filename = os.path.join(save_dir, f'heatmap_{title.replace(" ", "_")}_patient_{patient_id}.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(mean_powers, annot=True, cmap='coolwarm', cbar_kws={'label': 'Power'}, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_difference_plots(data, title, filename):
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    left_data = data[data['Hemisphere'] == 'HemisphereLocationDef.Left']
    right_data = data[data['Hemisphere'] == 'HemisphereLocationDef.Right']
    differences = {band: left_data[band].mean() - right_data[band].mean() for band in bands}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(differences.keys(), differences.values(), color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Frequency Bands')
    ax.set_ylabel('Difference (Left - Right)')
    plt.tight_layout()
    save_plot(fig, filename)

def plot_full_spectrum(data, title, filename):
    fig, ax = plt.subplots(figsize=(15, 8))
    band_edges = [0, 4, 8, 13, 30, 100]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    band_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'pink']

    for hemisphere in ['HemisphereLocationDef.Left', 'HemisphereLocationDef.Right']:
        hemisphere_data = data[data['Hemisphere'] == hemisphere]
        mean_powers = hemisphere_data[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].mean()
        freq_centers = [2, 6, 10.5, 21.5, 65]
        ax.plot(freq_centers, mean_powers, label=f'{hemisphere.split(".")[-1]} Hemisphere', marker='o')

    for i, band in enumerate(band_names):
        ax.axvspan(band_edges[i], band_edges[i + 1], color=band_colors[i], alpha=0.2, label=band)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xticks(freq_centers)
    ax.set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
    plt.tight_layout()
    save_plot(fig, filename)

def plot_grouped_full_spectrum(data, group_column, title, filename):
    fig, ax = plt.subplots(figsize=(15, 8))
    for group in data[group_column].unique():
        group_data = data[data[group_column] == group]
        mean_powers = group_data[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].mean()
        freq_centers = [2, 6, 10.5, 21.5, 65]
        ax.plot(freq_centers, mean_powers, label=group, marker='o')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xticks(freq_centers)
    ax.set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
    plt.tight_layout()
    save_plot(fig, filename)

def plot_power_vs_frequency_spectral_analysis(data, patient_id, symptom, output_dir, window_length=11, polyorder=2):
    def filter_extremes(df):
        if not df.empty:
            min_power = df['Power'].min()
            max_power = df['Power'].max()
            return df[(df['Power'] > min_power) & (df['Power'] < max_power)]
        return df

    def smooth_data(df):
        if not df.empty:
            df['Power'] = savgol_filter(df['Power'], window_length=window_length, polyorder=polyorder)
        return df

    left_data = smooth_data(filter_extremes(data[data['Hemisphere'] == 'HemisphereLocationDef.Left'].sort_values('Frequency')))
    right_data = smooth_data(filter_extremes(data[data['Hemisphere'] == 'HemisphereLocationDef.Right'].sort_values('Frequency')))

    patient_dir = os.path.join(output_dir, f'patient_{patient_id}', symptom)
    ensure_directory_exists(patient_dir)
    filename = os.path.join(patient_dir, f'power_vs_frequency_spectral_analysis_{symptom}.png')

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(left_data['Frequency'], left_data['Power'], label='Left Hemisphere', color='blue', alpha=0.7, linewidth=2)
    ax.plot(right_data['Frequency'], right_data['Power'], label='Right Hemisphere', color='orange', alpha=0.7, linewidth=2)

    bands = {
        'Delta': (0, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    band_colors = ['blue', 'green', 'yellow', 'red', 'purple']

    for (band, (start, end)), color in zip(bands.items(), band_colors):
        ax.axvspan(start, end, color=color, alpha=0.2, label=f'{band} Band')
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.85, band, ha='center', va='center', fontsize=12, color=color)

    ax.set_title(f'Power vs. Frequency - {symptom} (Patient {patient_id})', fontsize=16)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Power', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout(pad=0.5)  # Adjust padding to reduce white space
    save_plot(fig, filename)

def plot_violin_per_symptom(data, output_dir, patient_id='all', title_suffix='', max_points=1000):

    """
    Plot violin plots of frequency bands for each symptom, showing the distribution of power values.

    Args:
        data (pd.DataFrame): Data containing the frequency bands and symptom information.
        output_dir (str): Directory where plots will be saved.
        title_suffix (str): Suffix to add to the plot titles and filenames.
        max_points (int): Maximum number of points for the swarm plot to avoid overloading.
    """
    ensure_directory_exists(output_dir)
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    for symptom in data['EventName'].unique():
        symptom_data = data[data['EventName'] == symptom]
        melted_data = symptom_data[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']].melt(var_name='Band', value_name='Power')

        # Downsample data if it's too large
        if len(melted_data) > max_points:
            melted_data = melted_data.sample(max_points, random_state=42)  # Randomly sample points for swarm plot

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.violinplot(
            x='Band',
            y='Power',
            data=melted_data,
            inner='quartile',
            palette=palette,
            scale='width',
            bw=0.2,
            cut=0,
            linewidth=1,
            ax=ax
        )

        # Only add swarm plot if data size is manageable
        if len(melted_data) <= max_points:
            sns.swarmplot(
                x='Band',
                y='Power',
                data=melted_data,
                color='k',
                alpha=0.5,
                size=3,
                ax=ax
            )

        ax.set_title(f'Violin Plot of Frequency Bands - {symptom} {title_suffix}', fontsize=16)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Power')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        plot_filename = f'violin_plot_{symptom.replace(" ", "_")}{title_suffix}.png'
        plt.tight_layout()
        save_plot(fig, os.path.join(output_dir, plot_filename))




    """
    Plot improved violin plots of frequency bands for each symptom, showing the distribution of power values.

    Args:
        data (pd.DataFrame): Data containing the frequency bands and symptom information.
        output_dir (str): Directory where plots will be saved.
        patient_id (str): Patient ID or 'all' for combined data.
        title_suffix (str): Suffix to add to the plot titles and filenames.
        max_points (int): Maximum number of points for the swarm plot to avoid overloading.
    """
    ensure_directory_exists(output_dir)
    sns.set_style("whitegrid")
    
    # Define consistent color palette and band order
    band_order = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    color_palette = sns.color_palette("husl", n_colors=len(band_order))
    band_color_dict = dict(zip(band_order, color_palette))

    for symptom in data['EventName'].unique():
        symptom_data = data[data['EventName'] == symptom]
        melted_data = symptom_data[band_order].melt(var_name='Band', value_name='Power')
        
        # Downsample data if it's too large
        if len(melted_data) > max_points:
            melted_data = melted_data.groupby('Band').apply(lambda x: x.sample(n=max_points // len(band_order), replace=True)).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot violins
        sns.violinplot(
            x='Band',
            y='Power',
            data=melted_data,
            order=band_order,
            palette=band_color_dict,
            inner="quartile",
            cut=0,
            scale="width",
            ax=ax
        )
        
        # Add swarm plot for individual data points
        sns.swarmplot(
            x='Band',
            y='Power',
            data=melted_data,
            order=band_order,
            color=".25",
            size=3,
            alpha=0.5,
            ax=ax
        )

        # Customize the plot
        ax.set_title(f'Frequency Band Power Distribution - {symptom} (Patient {patient_id})', fontsize=16)
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add subtle grid lines
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_filename = f'violin_plot_{symptom.replace(" ", "_")}_patient_{patient_id}{title_suffix}.png'
        save_plot(fig, os.path.join(output_dir, plot_filename))
        plt.close(fig)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_correlation_heatmap(correlation_results, output_dir):
    """
    Create a heatmap of correlations between LFP and symptoms for each patient.
    
    :param correlation_results: DataFrame with correlation results
    :param output_dir: Directory to save the output plot
    """
    pivot_data = correlation_results.pivot(index='Symptom', columns='PatientID', values='Correlation')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('LFP-Symptom Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}lfp_symptom_correlation_heatmap.png')
    plt.close()

def visualize_correlation_barplot(correlation_results, output_dir):
    """
    Create a bar plot of correlations between LFP and symptoms for each patient.
    
    :param correlation_results: DataFrame with correlation results
    :param output_dir: Directory to save the output plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Symptom', y='Correlation', hue='PatientID', data=correlation_results)
    plt.title('LFP-Symptom Correlations by Patient')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Patient ID')
    plt.tight_layout()
    plt.savefig(f'{output_dir}lfp_symptom_correlation_barplot.png')
    plt.close()

def visualize_p_value_scatterplot(correlation_results, output_dir):
    """
    Create a scatter plot of correlations vs p-values.
    
    :param correlation_results: DataFrame with correlation results
    :param output_dir: Directory to save the output plot
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Correlation', y='P-value', hue='PatientID', style='Symptom', data=correlation_results)
    plt.title('Correlation vs P-value')
    plt.axhline(y=0.05, color='r', linestyle='--')  # Add significance threshold line
    plt.text(0.95, 0.05, 'p=0.05', verticalalignment='bottom', horizontalalignment='right', color='r')
    plt.tight_layout()
    plt.savefig(f'{output_dir}correlation_vs_pvalue_scatterplot.png')
    plt.close()

def visualize_effect_size_barplot(correlation_results, output_dir):
    """
    Create a bar plot of effect sizes for each symptom and patient.
    
    :param correlation_results: DataFrame with correlation results
    :param output_dir: Directory to save the output plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Symptom', y='Effect_Size', hue='PatientID', data=correlation_results)
    plt.title('Effect Size of LFP-Symptom Relationships')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Patient ID')
    plt.tight_layout()
    plt.savefig(f'{output_dir}effect_size_barplot.png')
    plt.close()

def create_all_visualizations(correlation_results, output_dir):
    """
    Create all visualizations for the LFP-symptom correlation results.
    
    :param correlation_results: DataFrame with correlation results
    :param output_dir: Directory to save the output plots
    """
    visualize_correlation_heatmap(correlation_results, output_dir)
    visualize_correlation_barplot(correlation_results, output_dir)
    visualize_p_value_scatterplot(correlation_results, output_dir)
    visualize_effect_size_barplot(correlation_results, output_dir)
    
    logging.info(f"All LFP-symptom correlation visualizations saved to {output_dir}")

# Usage in main script:
# create_all_visualizations(correlation_results, VISUALIZATIONS_DIR)