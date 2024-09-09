# LFP-Hero

**LFP-Hero** is a Python project designed to analyze Local Field Potential (LFP) data collected from Medtronic Percept™ PC Neurostimulator devices. This tool processes LFP data, performs statistical analyses, and generates insightful visualizations to identify potential biomarkers for Parkinson’s disease or other pathologies.

## Project Overview

The project is structured to modularly handle data processing, analysis, and visualization tasks, providing a comprehensive approach to understanding the LFP data collected during Deep Brain Stimulation (DBS) procedures. The analysis focuses on identifying patterns within frequency bands, comparing hemispheric differences, and visualizing trends over time.

## Features

- **Data Processing**: Extracts snapshot and trend data from Medtronic Percept™ JSON files.
- **Statistical Analysis**: Performs statistical comparisons of power between hemispheres and identifies the strongest frequency bands.
- **Correlation Studies**: Analyzes the correlation between LFP signals and specific frequency bands.
- **Visualization**: Generates various plots, including heatmaps, time series, and frequency spectra, to aid in data interpretation.

## Data Format

The data used in this project is sourced from Medtronic Percept™ PC Neurostimulator JSON files. For more information on the data format, refer to the [Medtronic Percept Whitepaper](https://www.medtronic.com/content/dam/medtronic-wide/public/western-europe/products/neurological/percept-pc-neurostimulator-whitepaper.pdf).

## Project Structure

```
LFP-Hero/
│
├── README.md                   # Project overview, setup, and usage instructions
├── LICENSE                     # License information
├── .gitignore                  # Specifies files and directories to be ignored by Git
├── requirements.txt            # List of dependencies for the project
├── data/                       # Directory for input data files
│   ├── json_report_patient1.json
│   └── json_report_patient2.json
│
├── output/                     # Directory for generated CSV files and visualizations
│   ├── statistical_results/
│   ├── visualizations/
│   └── combined_snapshot_data.csv
│
├── src/                        # Source code directory
│   ├── __init__.py             # Marks the directory as a package
│   ├── data_processing.py      # Data loading and processing functions
│   ├── analysis.py             # Functions for statistical analysis
│   ├── visualization.py        # Plotting and visualization functions
│   └── utils.py                # Utility functions, constants, and configurations
│
└── main.py                     # Main script to run the analysis
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/LFP-Hero.git
   cd LFP-Hero
   ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your Medtronic Percept™ JSON files in the `data/` directory.
2. Run the main analysis script:

   ```bash
   python main.py
   ```

3. The results, including statistical summaries and visualizations, will be saved in the `output/` directory.

## Output

- **Combined Snapshot Data**: Aggregated CSV of all processed snapshot data.
- **Statistical Results**: CSV files containing statistical analysis results.
- **Visualizations**: PNG images of various plots, including heatmaps, time series, and frequency spectra.

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and open pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Medtronic Percept Whitepaper](https://www.medtronic.com/content/dam/medtronic-wide/public/western-europe/products/neurological/percept-pc-neurostimulator-whitepaper.pdf)
