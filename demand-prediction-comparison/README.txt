Place demand_comparison/, ridge_demand/ and hgbr_demand/ beside each other in the repository root.
Run from the repository root: python demand_comparison/plot_demand_comparison.py
The script resolves input paths relative to its own parent directory, independent of your working directory. Outputs default to the script folder. No ZIP input is required.
Window: 6–12 February 2015, 336 half-hours, interval-ending AEST.
Ridge: 48 slot-specific models, HDD/CDD 17/19.5 C, alpha=10. HGBR: selected frozen predictions. Both trained through 2014.
Optional: --start YYYY-MM-DD --end YYYY-MM-DD --out PATH
