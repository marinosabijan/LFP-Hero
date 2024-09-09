# main.py
import os
import pandas as pd
import logging
from config import (
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    DATA_DIR,
    OUTPUT_DIR,
    COMBINED_SNAPSHOT_DATA_CSV,
    COMBINED_LFP_TREND_DATA_CSV,
    STRONGEST_BAND_TABLE_CSV,
    MEAN_POWER_TABLE_CSV,
    STATISTICAL_SUMMARY_CSV,
    VISUALIZATIONS_DIR,
    BAND_COUPLING_DIR,
)
from src.data_processing import load_json_data, extract_snapshot_data, extract_lfp_trend_data
from src.analysis import (
    perform_statistical_analysis,
    analyze_lfp_frequency_correlation,
    generate_strongest_band_table,
    generate_mean_power_table,
    generate_statistical_summary,
    check_timestamp_issues,
    perform_band_coupling_analysis,
    calculate_lfp_symptom_correlation,
    analyze_shared_events,
    interpret_results,
)
from src.visualization import (
    plot_power_vs_frequency_spectral_analysis,
    plot_lfp_trend_with_events,
    plot_heatmap,
    plot_difference_plots,
    plot_full_spectrum,
    plot_grouped_full_spectrum,
    plot_violin_per_symptom,
    create_all_visualizations,
)

# Configure logging
logging.basicConfig(
    level=LOGGING_LEVEL, 
    format=LOGGING_FORMAT,
    handlers=[logging.StreamHandler()]
)

def main():
    logging.info("Starting main analysis")

    # Load data for both patients
    patient1_data = load_json_data(f'{DATA_DIR}json_report_patient1.json')
    patient2_data = load_json_data(f'{DATA_DIR}json_report_patient2.json')

    # Extract snapshot data for each patient
    patient1_snapshot_data = extract_snapshot_data(patient1_data, patient_id=1)
    patient2_snapshot_data = extract_snapshot_data(patient2_data, patient_id=2)

    # Extract LFP trend data for each patient
    patient1_lfp_trend_data = extract_lfp_trend_data(patient1_data, patient_id=1)
    patient2_lfp_trend_data = extract_lfp_trend_data(patient2_data, patient_id=2)

    # Combine snapshot data for both patients
    combined_snapshot_data = pd.concat([patient1_snapshot_data, patient2_snapshot_data], ignore_index=True)
    
    # Combine LFP trend data for both patients
    combined_lfp_trend_data = pd.concat([patient1_lfp_trend_data, patient2_lfp_trend_data], ignore_index=True)
    
    # Convert 'DateTime' to datetime format
    combined_snapshot_data['DateTime'] = pd.to_datetime(combined_snapshot_data['DateTime'])
    combined_lfp_trend_data['DateTime'] = pd.to_datetime(combined_lfp_trend_data['DateTime'])

    # Check for timestamp issues
    check_timestamp_issues(combined_snapshot_data)
    check_timestamp_issues(combined_lfp_trend_data)

    # Save combined data
    combined_snapshot_data.to_csv(COMBINED_SNAPSHOT_DATA_CSV, index=False)
    logging.info(f'Data saved to {COMBINED_SNAPSHOT_DATA_CSV}')

    combined_lfp_trend_data.to_csv(COMBINED_LFP_TREND_DATA_CSV, index=False)
    logging.info(f'Data saved to {COMBINED_LFP_TREND_DATA_CSV}')

    # Perform correlation analysis
    correlation_results = analyze_lfp_frequency_correlation(combined_snapshot_data, combined_lfp_trend_data)

    # Generate tables
    strongest_band_table = generate_strongest_band_table(combined_snapshot_data)
    strongest_band_table.to_csv(STRONGEST_BAND_TABLE_CSV, index=False)
    logging.info(f'Data saved to {STRONGEST_BAND_TABLE_CSV}')

    mean_power_table = generate_mean_power_table(combined_snapshot_data)
    mean_power_table.to_csv(MEAN_POWER_TABLE_CSV, index=False)
    logging.info(f'Data saved to {MEAN_POWER_TABLE_CSV}')

    # Perform statistical analysis for all data
    statistical_results = {}
    for symptom in combined_snapshot_data['EventName'].unique():
        symptom_data = combined_snapshot_data[combined_snapshot_data['EventName'] == symptom]
        statistical_results[symptom] = perform_statistical_analysis(symptom_data)

    # Generate statistical summary
    statistical_summary = generate_statistical_summary(statistical_results)
    statistical_summary.to_csv(STATISTICAL_SUMMARY_CSV, index=False)
    logging.info(f'Statistical summary saved to {STATISTICAL_SUMMARY_CSV}')
    
    # Generate plots for each patient and symptom
    for patient_id in combined_snapshot_data['PatientID'].unique():
        patient_data = combined_snapshot_data[combined_snapshot_data['PatientID'] == patient_id]
        patient_lfp_data = combined_lfp_trend_data[combined_lfp_trend_data['PatientID'] == patient_id]

        # Create a directory for each patient's band coupling results
        patient_band_coupling_dir = os.path.join(BAND_COUPLING_DIR, f'patient_{patient_id}')
        os.makedirs(patient_band_coupling_dir, exist_ok=True)

        # Generate LFP trend charts with mapped event snapshots
        plot_lfp_trend_with_events(
            patient_lfp_data,
            patient_data,
            patient_id,
            VISUALIZATIONS_DIR
        )

        # Generate violin and spectral plots for each symptom
        for symptom in patient_data['EventName'].unique():
            symptom_data = patient_data[patient_data['EventName'] == symptom]
            plot_power_vs_frequency_spectral_analysis(symptom_data, patient_id, symptom, VISUALIZATIONS_DIR)
            plot_violin_per_symptom(symptom_data, VISUALIZATIONS_DIR, patient_id=patient_id, title_suffix=f'_patient_{patient_id}')

            # Perform band coupling analysis for each symptom
            symptom_band_coupling_dir = os.path.join(patient_band_coupling_dir, symptom)
            os.makedirs(symptom_band_coupling_dir, exist_ok=True)
            
            try:
                band_coupling_results = perform_band_coupling_analysis(symptom_data, symptom_band_coupling_dir)
                logging.info(f"Band coupling analysis completed for Patient {patient_id}, Symptom: {symptom}")
            except Exception as e:
                logging.error(f"Error in band coupling analysis for Patient {patient_id}, Symptom: {symptom}: {str(e)}")
                
        # Combined plots for all symptoms per patient
        plot_grouped_full_spectrum(patient_data, 'EventName', 
                                   f'Full Spectrum Analysis - Patient {patient_id}', 
                                   os.path.join(VISUALIZATIONS_DIR, f'grouped_patient_{patient_id}.png'))

    # Generate grouped plots for combined data of both patients
    plot_grouped_full_spectrum(combined_snapshot_data, 'EventName', 
                               'Full Spectrum Analysis - All Patients', 
                               os.path.join(VISUALIZATIONS_DIR, 'grouped_all_patients.png'))

    # Generate heatmaps and LFP trend plots for all patients combined
    plot_heatmap(combined_snapshot_data, 'Mean Power per Event', 'all_patients', VISUALIZATIONS_DIR)
    plot_lfp_trend_with_events(combined_lfp_trend_data, combined_snapshot_data, 'all', VISUALIZATIONS_DIR)

    # New additions: LFP-Symptom Correlation Analysis
    correlation_results = calculate_lfp_symptom_correlation(combined_lfp_trend_data, combined_snapshot_data)
    correlation_results.to_csv(f'{OUTPUT_DIR}lfp_symptom_correlation_detailed.csv', index=False)
    logging.info(f"Detailed LFP-symptom correlation results saved to {OUTPUT_DIR}lfp_symptom_correlation_detailed.csv")

    # Create visualizations for LFP-Symptom Correlation results
    create_all_visualizations(correlation_results, VISUALIZATIONS_DIR)

    # Perform shared event analysis
    shared_events = ["Dobro sam"]  # Add other shared events as needed
    shared_events_analysis = analyze_shared_events(correlation_results, shared_events)
    shared_events_analysis.to_csv(f'{OUTPUT_DIR}shared_events_analysis.csv', index=False)
    logging.info(f"Shared events analysis saved to {OUTPUT_DIR}shared_events_analysis.csv")

    # Generate interpretation of results
    interpretation = interpret_results(correlation_results, shared_events_analysis)
    with open(f'{OUTPUT_DIR}correlation_analysis_interpretation.txt', 'w') as f:
        f.write(interpretation)
    logging.info(f"Correlation analysis interpretation saved to {OUTPUT_DIR}correlation_analysis_interpretation.txt")

    logging.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()