# 🌍 Downscale Anything RF Fast (QGIS Plugin)

[![QGIS](https://img.shields.io/badge/QGIS-%E2%89%A53.0-green)](https://qgis.org)  
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Build](https://img.shields.io/badge/status-stable-success)]()  
[![Author](https://img.shields.io/badge/Author-Firman%20Afrianto-lightgrey)]()  

---

## 📖 Overview
**Downscale Anything RF Fast** is a QGIS plugin for **machine-learning-based raster downscaling** using **Random Forest regression**.  
It enables redistribution of coarse raster data into higher-resolution grids with the help of fine-scale covariates, based on the principle of **dasymetric mapping**.

> ✨ Designed for spatial analysis, environmental modeling, and urban/regional planning.

---

## ⚙️ Features
- ✅ Downscale any coarse raster (VIIRS, LST, WorldPop, etc.) to **user-defined resolution** (e.g., 30m, 10m, even 1m).  
- ✅ Random Forest regression with **fast random pixel sampling** for efficient training.  
- ✅ Automatic evaluation with **R², RMSE, MAE, and OOB score**.  
- ✅ Multiple outputs:
  - 🗺️ **GeoTIFF** — downscaled raster  
  - 📊 **CSV** — evaluation metrics + feature importance  
  - 🖼️ **PNG** — scatterplot, residual histogram, feature importance  

---

## 📂 Example Use Cases
- 🌃 **VIIRS Nighttime Lights (500m → 30m)** for urbanization mapping.  
- 🌡️ **Land Surface Temperature (1km → 100m)** for Urban Heat Island studies.  
- 👥 **WorldPop Population Density (1km → 30m)** for vulnerability/exposure analysis.  
- 🍃 **PM2.5 / Air Pollution (1km → 100m)** for environmental health monitoring.  
- 🌧️ **Precipitation (5km → 500m)** for hydrological and flood modeling.  

---

## 🖥️ Outputs
The plugin generates three types of outputs:  

| Output Type      | Description |
|------------------|-------------|
| 🗺️ **GeoTIFF**   | Downscaled raster at target resolution |
| 📊 **CSV**       | Evaluation results and feature importance |
| 🖼️ **PNG**       | Scatterplot, residual histogram, feature importance |

---

## 🚀 Installation
1. Clone or download this repository.  
2. Place the folder in your **QGIS plugins directory**:  
   - Windows: `%AppData%\QGIS\QGIS3\profiles\default\python\plugins\`  
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`  
3. Restart QGIS and enable the plugin via **Plugin Manager**.  
4. Install Python dependencies:  
   ```bash
   pip install scikit-learn pandas rasterio matplotlib
