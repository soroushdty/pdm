# Physician Decision-Making Simulator

1. Purpose
----------

This project provides a modular data processing pipeline for automated patient sensitive data segmentation using public LLMs.
The pipeline:
    • Cleans and merges annotations
    • Standardizes item names using manual or automated matching
    • Prevents train/test data leakage
    • Produces reproducible, logged outputs

The pipeline is designed to run end-to-end from a single notebook while keeping all logic in reusable Python modules.

2. How the Code Is Organized
----------------------------

.
├── main.ipynb                  # Entry point (run this)
├── requirements.txt            # Dependency list
├── pdm_preprocess/
│   ├── __init__.py
│   ├── requirements_utils.py   # Dependency handling
│   ├── config_loader.py        # Configuration loading
│   ├── paths.py                # Output directory management
│   ├── logger_setup.py         # Logging configuration
│   ├── io_dataset.py           # Dataset I/O
│   ├── merge_physician_rows.py # Merge duplicate annotations
│   ├── item_standardization.py # Automatic item standardization
│   ├── corrections.py          # Manual correction application
│   ├── leakage.py              # Train/test leakage prevention
│   └── pipeline.py             # Pipeline orchestration

3. Execution Flow
-----------------

The execution starts in main.ipynb and proceeds as follows:
    1. Install and import dependencies
    2. Load and validate configuration
    3. Create output directories and logger
    4. Load datasets
    5. Run preprocessing pipeline
    6. Save results and logs

All heavy logic is handled inside Python modules.

4. Input
--------

4.1 Configuration File (JSON)
The pipeline is controlled by a JSON configuration file.

Required fields:
    {
      "DIR_INPUT": "/path/to/dataset.xlsx",
      "output_path": "/path/to/output_directory"
    }

Optional fields (defaults applied):
    {
      "patient_col": "Patient",
      "physician_col": "Physician",
      "item_col": "Item",
      "classes": ["Class1", "Class2", "Class3"],
      "train_sheet": "train",
      "test_sheet": "test"
    }

4.2 Dataset Format
    • Excel or CSV file
    • Train and test data are expected as separate sheets (or files)
    • Each row represents a physician annotation
    • Multiple physicians may annotate the same patient–item pair

5. Output
---------

All outputs are written to a timestamped run directory:
    output_path/
        run_YYYYMMDD_HHMMSS/

Generated files include:
    • Cleaned train dataset
    • Cleaned test dataset
    • JSON file containing standardized item mappings
    • Log file describing each pipeline step

6. Modules Description
----------------------

requirements_utils.py
    • Reads requirements.txt
    • Checks which packages are available
    • Installs missing dependencies
    • Imports all required modules dynamically

config_loader.py
    • Loads the JSON configuration file
    • Validates required fields
    • Applies default values
    • Returns a normalized configuration dictionary

paths.py
    • Creates the main output directory
    • Creates a unique run subdirectory
    • Returns all relevant filesystem paths

logger_setup.py
    • Configures logging to console and file
    • Ensures consistent formatting
    • Used by all modules

io_dataset.py
    • Loads train and test datasets
    • Enforces column typing
    • Handles Excel and CSV formats

merge_physician_rows.py
    • Merges rows with identical (Patient, Item)
    • Aggregates multiple physician annotations
    • Averages class-related numeric columns

item_standardization.py
    • Automatically groups similar item names
    • Uses fuzzy string matching
    • Produces a JSON mapping of standardized items
    • Designed for manual review if needed

corrections.py
    • Applies item corrections from a JSON mapping
    • Updates item names deterministically
    • Re-merges rows after corrections

leakage.py
    • Detects overlapping standardized items
    • Removes overlaps between train and test sets
    • Prevents information leakage

pipeline.py
    • Orchestrates all processing steps
    • Ensures correct order of operations
    • Handles intermediate and final outputs

7. How to Run (Google Colab)
----------------------------

    1. Upload:
        - pdm_preprocess/
        - requirements.txt
        - main.ipynb
        - Configuration JSON
    2. Open main.ipynb
    3. Set:

       CONFIG_PATH = "/content/config.json"

    4. Run all cells

8. Design Principles
--------------------

    • Modular and reusable
    • Configuration-driven
    • Fully logged
    • Reproducible outputs
    • Safe to re-run multiple times
    • Minimal logic inside notebooks

9. Extensibility
----------------

The modular structure allows:
    • Replacing fuzzy matching with embeddings
    • Adding validation layers
    • Writing unit tests per module
    • Running individual pipeline steps independently

10. Notes
---------

    • Automatic item standardization should be reviewed before downstream use
    • Large datasets may increase runtime during fuzzy matching
