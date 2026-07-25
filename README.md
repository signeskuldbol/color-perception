# color-perception
This project consists of 2 files:
    create_excel_files_and_sort_data
        - Creates excel files: one containing all data, and another for the discarded attempts with reasoning  
    analyse_and_plot_data
        - Creates data plots and performs data analasys

## Running the Code
Start by opening "create_excel_files_and_sort_data" search for "TODO" and update the folder name for the raw data folder. 

Expected project structure:
    color-perception/
    └── data_color_crush/
        ├── create_excel_files_and_sort_data
        ├── "-Folder Name-"/
        │   ├── participant_file_1.txt
        │   ├── participant_file_2.txt
        │   └── ...
        └── excel_files/
            ├── combined_final_results.xlsx
            └── discarded_results.xlsx

By running the script the following is created:
    1. combined_final_results.xlsx
        Sheets:
            - participant_summary: one row per participant with demographics and level summary (highest level overview)
            - all_levels_short: one row per participant per level with summary info 
            - all_final_colors_long: all data rows for all participants and all levels' sublevels.
            - level_1 to level_8: one sheet per level with all kept attempts for that level

    2. discarded_results.xlsx
        Sheets:
        - empty_files: participant files with no log rows found
        - discarded_level_attempts: structurally incomplete attempts or attempts removed due to filtering.

Then open "analyse_and_plot_data" and run that. This creates plots for the data and performs analasys of the data. 
