# Data

All data are public. Prices are converted to rupees per 40 kg, the unit used in Pakistan's wheat trade. Please cite the original sources (listed below and in the paper) when you use any of these files.

## `raw/`: source files included in this repository

| File | Source | License | Used for |
|---|---|---|---|
| `wfp_food_prices_pak.csv` | World Food Programme, *Pakistan: Food prices*, via the Humanitarian Data Exchange: https://data.humdata.org/dataset/wfp-food-prices-for-pakistan (downloaded November 2026) | Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO), as for WFP's food-price data on HDX | Primary panel: wheat grain prices in Lahore, Multan, Karachi, Peshawar and Quetta |
| `CMO-Historical-Data-Monthly.xlsx` | World Bank, *Commodity price data (The Pink Sheet), monthly historical prices*: https://www.worldbank.org/en/research/commodity-markets (version updated November 3, 2026) | CC BY 4.0 (World Bank open data) | US hard red winter wheat price |
| `USD_To_PKR_1992_kaggle.csv` | Baig, S. U. (2023). *Currency rate USD to PKR (Feb 1992 to Nov 2023)* (Version 1) [Data set]. Kaggle. https://www.kaggle.com/datasets/samiullahbaig/currency-rate-usd-to-pkr-feb-1992-to-nov-2023 | Apache 2.0 | Daily rupees per US dollar before July 2015. Version 1 (November 2023) covers February 1992 to November 2023; the dataset was later extended and retitled |

## Source files not included

The terms for redistributing these files could not be confirmed, so they are not stored here. The panels built from them are in `derived/`, so they are needed only to rebuild the panels (`REBUILD_PANELS=1 bash run_all.sh`). Download them into `raw/` under the names given.

| Save as | Source | Used for |
|---|---|---|
| `Policy_Rate.csv` | State Bank of Pakistan, EasyData, policy rate series: https://easydata.sbp.org.pk | Cost of capital |
| `Exchange_Rates_-_PKR_US_.csv` | State Bank of Pakistan, EasyData, PKR per US dollar (daily): https://easydata.sbp.org.pk | Rupees per US dollar from July 2015 |
| `CpI-Urban-Groupwise-Cumulative-Indices.pdf`, `RPI-Gropwise-Cumulative-Indices.pdf`, `indices_and_growth_rates_historical-1.pdf` | Pakistan Bureau of Statistics, consumer price index tables: https://www.pbs.gov.pk/cpi | Real-terms check |
| `fao_fpma_pakistan_wheat_flour_retail.xlsx` | FAO GIEWS Food Price Monitoring and Analysis Tool: https://fpma.fao.org/giews/fpmat4/ — export Pakistan, retail, wheat (flour), PKR/kg for Lahore, Multan, Karachi, Peshawar and Quetta (columns `iso3_country_code`, `Date`, then one column per city named like `Pakistan, RETAIL, Lahore, Wheat (flour), PKR/Kg`) | Robustness panel |
| `ASP_for_Web_2024.pdf` | Ministry of National Food Security and Research, *Agricultural Statistics of Pakistan 2023–24*: https://mnfsr.gov.pk/SiteImage/Misc/files/ASP%20for%20Web%202024.pdf | Validation panel (Table 145, July 2006 to June 2024) |
| `ASP_2024-25_FINAL.pdf` | Ministry of National Food Security and Research, *Agricultural Statistics of Pakistan 2024–25*: https://mnfsr.gov.pk/SiteImage/Publication/ASP%202024-25%20FINAL.pdf | Validation panel (July 2024 to June 2025) |

## `compiled/`

| File | Contents |
|---|---|
| `support_prices.csv` | Wheat support price (rupees per 40 kg) in force at each harvest, 2006–2026, and whether Punjab procured. Compiled from the sources named in the `source` column (Agricultural Statistics of Pakistan, published studies and news reports; see Appendix Table E1 of the paper) |

## `derived/`: built by the scripts in `code/`

| File | Built by | Contents |
|---|---|---|
| `monthly_panel.csv` | `build_monthly_panel.py` | One row per month: WFP wheat prices, exchange rate (`fx_source` says SBP or Kaggle), world wheat prices, price indices, policy rate, world price in rupees per 40 kg and the world-parity gap |
| `cpi_urban_general.csv` | `build_monthly_panel.py` | Urban CPI, general index, as printed by PBS (2007-08 = 100 to June 2019, 2015-16 = 100 after) |
| `pbs_wholesale_wheat.csv` | `parse_pbs_wholesale.py` | Official monthly wholesale wheat prices, six markets, from Table 145 of the Agricultural Statistics of Pakistan. In the 2024–25 edition the 2024–25 row is about 2.5 times the earlier level although the header says rupees per 40 kg; it matches rupees per 100 kg, so it is divided by 2.5 (flagged in the `note` column; see Appendix E of the paper) |
| `colab_panel.csv` | `build_colab_panel.py` | The 17 series used by the models (`unique_id`, `ds`, `y` in rupees per 40 kg): WFP grain (5 markets), official wholesale (6 markets), FAO flour (5 cities) and the Lahore–Multan average, which is used only in figures and as a training series |
| `panel_flags.csv` | `build_colab_panel.py` | Months filled forward over gaps of up to three months; these are never used as a forecast origin or target |
| `panel_series_info.csv` | `build_colab_panel.py` | Start, end and length of each series |

Series names starting with `pbs_whsl_` are the official wholesale prices (data from the provincial marketing departments and PBS, published in the Agricultural Statistics of Pakistan).
