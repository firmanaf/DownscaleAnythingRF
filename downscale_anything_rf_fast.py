# -*- coding: utf-8 -*-
"""
Downscale Anything with Random Forest — QGIS Processing Toolbox (Fast Sampling)
Generic dasymetric downscaling from a coarse raster to user-defined resolution
using high-resolution covariates and Random Forest regression.

Features:
- User-specified target resolution
- Per-layer NoData handling
- Categorical resampling detection
- Safe CRS handoff to GDAL
- Optional chunked prediction to reduce memory
- Spatial-block split and balanced sampling
- Optional nonnegative clipping
- PNG includes scatter plot, residual histogram, and feature importance
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterFileDestination
)

import os
import math
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst


class DownscaleAnythingRFFast(QgsProcessingAlgorithm):

    INPUT_TARGET = 'INPUT_TARGET'
    INPUT_COVARS = 'INPUT_COVARS'
    TARGET_RES = 'TARGET_RES'
    NODATA = 'NODATA'
    MAX_SAMPLES = 'MAX_SAMPLES'
    TEST_SIZE = 'TEST_SIZE'
    SPLIT_MODE = 'SPLIT_MODE'      # 0 Random, 1 Spatial-block
    BLOCK_PX = 'BLOCK_PX'
    CHUNK_PX = 'CHUNK_PX'          # 0 full in memory, >0 tile side length in pixels
    CLIP_NONNEG = 'CLIP_NONNEG'    # 1 on, 0 off
    OUTPUT_RASTER = 'OUTPUT_RASTER'
    OUTPUT_CSV = 'OUTPUT_CSV'
    OUTPUT_PLOT = 'OUTPUT_PLOT'

    def tr(self, text):
        return QCoreApplication.translate('Processing', text)

    def createInstance(self):
        return DownscaleAnythingRFFast()

    def name(self):
        return 'downscale_anything_rf_fast'

    def displayName(self):
        return self.tr('Downscale Anything with Random Forest')

    def shortHelpString(self):
        return self.tr(
            "<p><b>Downscale Anything with Random Forest</b></p>"
            "<p>Downscale a coarse raster to a <b>user-defined resolution</b> using "
            "Random Forest regression and <i>high-resolution covariates</i>. "
            "Treat the result as <b>dasymetric redistribution</b> (pattern sharpening), "
            "not optical super-resolution.</p>"

            "<p><b>What it does</b></p>"
            "<ul>"
            "<li>Resamples target and covariates to the chosen grid.</li>"
            "<li>Trains a Random Forest on sampled pixels (fast), with optional spatial block split.</li>"
            "<li>Predicts the fine-grid surface; optional non-negative clipping.</li>"
            "</ul>"

            "<p><b>Inputs</b></p>"
            "<ul>"
            "<li><b>Target raster</b> (coarse; e.g., VIIRS NTL, LST, population).</li>"
            "<li><b>Covariates</b> (30 m or finer; e.g., Sentinel indices, elevation, roads, built-up).</li>"
            "<li><b>Target Resolution</b> (meters), <b>CHUNK_PX</b> for memory-safe tiling, "
            "<b>BLOCK_PX</b> for spatial split, and sampling size.</li>"
            "</ul>"

            "<p><b>Outputs</b></p>"
            "<ul>"
            "<li><b>Downscaled raster</b> at target resolution.</li>"
            "<li><b>CSV</b> with metrics (R², RMSE, MAE, OOB) and feature importance.</li>"
            "<li><b>PNG (3 panels)</b>: Observed vs Predicted scatter, Residual histogram, Feature importance.</li>"
            "</ul>"

            "<p><b>Good practice</b></p>"
            "<ul>"
            "<li>Use <b>spatial block split</b> for realistic validation.</li>"
            "<li>Prefer <b>100–150 m</b> as a safe step-down from ~500 m; "
            "use <b>30 m</b> or finer only if covariates are strong and interpretation is <i>relative</i>.</li>"
            "<li>Set <b>CHUNK_PX</b> (e.g., 512–1024) for large rasters to avoid out-of-memory.</li>"
            "</ul>"

            "<p><i>Created by</i> <b>FIRMAN AFRIANTO</b></p>"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_TARGET, self.tr('Target raster to downscale (coarse resolution)')))
        self.addParameter(QgsProcessingParameterMultipleLayers(
            self.INPUT_COVARS, self.tr('Covariate rasters at or finer than target resolution'),
            layerType=QgsProcessing.TypeRaster))
        self.addParameter(QgsProcessingParameterNumber(
            self.TARGET_RES, self.tr('Target resolution in meters'), type=QgsProcessingParameterNumber.Integer,
            defaultValue=30))
        self.addParameter(QgsProcessingParameterNumber(
            self.NODATA, self.tr('Fallback NoData if metadata missing'),
            type=QgsProcessingParameterNumber.Double, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_SAMPLES, self.tr('Maximum training samples'),
            type=QgsProcessingParameterNumber.Integer, defaultValue=10000))
        self.addParameter(QgsProcessingParameterNumber(
            self.TEST_SIZE, self.tr('Test set proportion'),
            type=QgsProcessingParameterNumber.Double, defaultValue=0.2))
        self.addParameter(QgsProcessingParameterNumber(
            self.SPLIT_MODE, self.tr('Split mode 0 Random 1 Spatial block'),
            type=QgsProcessingParameterNumber.Integer, defaultValue=1))
        self.addParameter(QgsProcessingParameterNumber(
            self.BLOCK_PX, self.tr('Block size in pixels for spatial split and balanced sampling'),
            type=QgsProcessingParameterNumber.Integer, defaultValue=128))
        self.addParameter(QgsProcessingParameterNumber(
            self.CHUNK_PX,
            self.tr('Prediction tile size in pixels. 0 means full in memory. '
                    'Small rasters under 5000x5000 use 0. Medium around 10k use 512 or 1024. '
                    'Very large use 256 or 512.'),
            type=QgsProcessingParameterNumber.Integer, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.CLIP_NONNEG, self.tr('Clip predictions to nonnegative 1 yes 0 no'),
            type=QgsProcessingParameterNumber.Integer, defaultValue=1))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_RASTER, self.tr('Output downscaled raster')))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_CSV, self.tr('Output evaluation CSV'), fileFilter='CSV files (*.csv)'))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_PLOT, self.tr('Output evaluation plot PNG'), fileFilter='PNG files (*.png)'))

    def _crs_to_srs(self, crs):
        try:
            return crs.to_wkt()
        except Exception:
            try:
                return crs.to_string()
            except Exception:
                return str(crs)

    def _is_categorical(self, dtype_str):
        return dtype_str.startswith('int') or dtype_str.startswith('uint')

    def processAlgorithm(self, parameters, context, feedback):

        target_res = int(self.parameterAsInt(parameters, self.TARGET_RES, context))
        nodata_fallback = self.parameterAsDouble(parameters, self.NODATA, context)
        max_samples = int(self.parameterAsInt(parameters, self.MAX_SAMPLES, context))
        test_size = float(self.parameterAsDouble(parameters, self.TEST_SIZE, context))
        split_mode = int(self.parameterAsInt(parameters, self.SPLIT_MODE, context))
        block_px = max(1, int(self.parameterAsInt(parameters, self.BLOCK_PX, context)))
        chunk_px = int(self.parameterAsInt(parameters, self.CHUNK_PX, context))
        clip_nonneg = int(self.parameterAsInt(parameters, self.CLIP_NONNEG, context)) == 1

        target_layer = self.parameterAsRasterLayer(parameters, self.INPUT_TARGET, context)
        covar_layers = self.parameterAsLayerList(parameters, self.INPUT_COVARS, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)
        output_plot = self.parameterAsFileOutput(parameters, self.OUTPUT_PLOT, context)

        rng = np.random.default_rng(42)
        temp_paths = []

        # Resample target to target_res
        feedback.pushInfo(f"Resampling target raster to {target_res} m")
        target_resamp = output_raster + "_target_resamp.tif"
        temp_paths.append(target_resamp)

        with rasterio.open(target_layer.source()) as src_in:
            srs = self._crs_to_srs(src_in.crs)

        gdal.Warp(
            target_resamp,
            target_layer.source(),
            xRes=target_res, yRes=target_res,
            resampleAlg=gdalconst.GRA_Bilinear,
            targetAlignedPixels=True,
            dstNodata=nodata_fallback,
            multithread=True,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER",
                             "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
            dstSRS=srs
        )

        with rasterio.open(target_resamp) as src:
            target_data = src.read(1)
            target_meta = src.meta.copy()
            target_bounds = src.bounds
            target_nodata = src.nodata if src.nodata is not None else nodata_fallback

        # Resample covariates to same grid
        covariates_list = []
        covariate_names = []
        covariate_nodata = []
        with rasterio.open(target_resamp) as tgt:
            dst_srs = self._crs_to_srs(tgt.crs)
            for cov in covar_layers:
                name = cov.name().replace(os.sep, "_").replace(" ", "_")
                feedback.pushInfo(f"Resampling {name} to {target_res} m")
                cov_resamp = output_raster + f"_{name}_resamp.tif"
                temp_paths.append(cov_resamp)

                with rasterio.open(cov.source()) as src_cov_raw:
                    dtype_str = str(src_cov_raw.dtypes[0])
                    res_alg = gdalconst.GRA_NearestNeighbour if self._is_categorical(dtype_str) else gdalconst.GRA_Bilinear

                gdal.Warp(
                    cov_resamp, cov.source(),
                    xRes=target_res, yRes=target_res,
                    resampleAlg=res_alg,
                    targetAlignedPixels=True,
                    outputBounds=[target_bounds.left, target_bounds.bottom,
                                  target_bounds.right, target_bounds.top],
                    dstSRS=dst_srs,
                    dstNodata=nodata_fallback,
                    multithread=True,
                    creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER",
                                     "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
                )
                with rasterio.open(cov_resamp) as csrc:
                    covariates_list.append(csrc.read(1))
                    covariate_names.append(name)
                    covariate_nodata.append(csrc.nodata if csrc.nodata is not None else nodata_fallback)

        # Build mask
        mask = np.isfinite(target_data)
        if target_nodata is not None:
            mask &= (target_data != target_nodata)
        for arr, nd in zip(covariates_list, covariate_nodata):
            m = np.isfinite(arr)
            if nd is not None:
                m &= (arr != nd)
            mask &= m

        rows, cols = np.where(mask)
        if rows.size == 0:
            raise ValueError("Empty mask after NoData filtering. Check inputs.")

        # Features and labels
        X_all = np.stack([arr[mask] for arr in covariates_list], axis=-1)
        y_all = target_data[mask]
        n = len(y_all)

        # Split
        if split_mode == 0:
            all_idx = np.arange(n)
            train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=42)
        else:
            tile_r = rows // block_px
            tile_c = cols // block_px
            groups = tile_r.astype(np.int64) * 100000 + tile_c.astype(np.int64)
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))

            # Balanced sampling across blocks for training set if very large
            if len(train_idx) > max_samples:
                train_groups = groups[train_idx]
                uniq = np.unique(train_groups)
                quota = max(1, int(math.ceil(max_samples / len(uniq))))
                keep = []
                for g in uniq:
                    idx_g = train_idx[train_groups == g]
                    if len(idx_g) > quota:
                        keep.append(rng.choice(idx_g, size=quota, replace=False))
                    else:
                        keep.append(idx_g)
                train_idx = np.concatenate(keep)

        if len(test_idx) > max_samples:
            test_idx = rng.choice(test_idx, size=max_samples, replace=False)

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

        feedback.pushInfo(f"Training samples {len(y_train)} Test samples {len(y_test)}")

        # Train RF with OOB
        rf = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1,
            oob_score=True, bootstrap=True
        )
        rf.fit(X_train, y_train)

        # Predict
        target_meta.update(dtype=rasterio.float32, count=1,
                           compress='DEFLATE', tiled=True, blockxsize=512, blockysize=512)
        height, width = target_data.shape
        default_nodata = target_nodata if target_nodata is not None else nodata_fallback
        predicted = np.full((height, width), default_nodata, dtype=np.float32)

        if chunk_px and chunk_px > 0:
            with rasterio.open(output_raster, "w", **target_meta) as dst:
                for r0 in range(0, height, chunk_px):
                    r1 = min(height, r0 + chunk_px)
                    for c0 in range(0, width, chunk_px):
                        c1 = min(width, c0 + chunk_px)
                        win = np.s_[r0:r1, c0:c1]
                        block_mask = mask[win]
                        if not block_mask.any():
                            dst.write(predicted[win], 1, window=((r0, r1), (c0, c1)))
                            continue
                        block_stack = np.stack([arr[win] for arr in covariates_list], axis=-1)
                        block_pred = np.full(block_mask.shape, default_nodata, dtype=np.float32)
                        block_pred[block_mask] = rf.predict(block_stack[block_mask]).astype(np.float32)
                        if clip_nonneg:
                            block_pred = np.clip(block_pred, 0, None)
                        predicted[win] = block_pred
                        dst.write(block_pred, 1, window=((r0, r1), (c0, c1)))
        else:
            full_stack = np.stack(covariates_list, axis=-1)
            predicted[mask] = rf.predict(full_stack[mask]).astype(np.float32)
            if clip_nonneg:
                predicted = np.clip(predicted, 0, None)
            with rasterio.open(output_raster, "w", **target_meta) as dst:
                dst.write(predicted, 1)

        # Evaluation metrics
        y_pred_train = rf.predict(X_train)
        y_pred_test  = rf.predict(X_test)
        r2_tr   = r2_score(y_train, y_pred_train)
        rmse_tr = math.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_tr  = mean_absolute_error(y_train, y_pred_train)
        r2_te   = r2_score(y_test, y_pred_test)
        rmse_te = math.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_te  = mean_absolute_error(y_test, y_pred_test)
        oob = getattr(rf, "oob_score_", None)

        # CSV sections
        importance_df = pd.DataFrame({"Covariate": covariate_names, "Importance": rf.feature_importances_}) \
                            .sort_values(by="Importance", ascending=False)
        eval_df = pd.DataFrame({
            "Set": ["Train", "Test", "OOB"],
            "R2":  [r2_tr, r2_te, oob if oob is not None else np.nan],
            "RMSE":[rmse_tr, rmse_te, np.nan],
            "MAE": [mae_tr, mae_te,  np.nan]
        })
        meta_df = pd.DataFrame({
            "Section": ["META", "META", "META", "META", "META", "META"],
            "Param": ["TARGET_RES", "MAX_SAMPLES", "TEST_SIZE", "SPLIT_MODE", "BLOCK_PX", "CHUNK_PX"],
            "Value": [target_res, max_samples, test_size, split_mode, block_px, chunk_px]
        })
        out_df = pd.concat([meta_df, pd.DataFrame({"Section": ["EVAL"]}), eval_df,
                            pd.DataFrame({"Section": ["IMPORTANCE"]}), importance_df], ignore_index=True)
        out_df.to_csv(output_csv, index=False)

        # PNG with three panels
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].scatter(y_test, y_pred_test, alpha=0.3, s=5)
        axs[0].set_xlabel("Observed Test")
        axs[0].set_ylabel("Predicted")
        axs[0].set_title(f"Downscale Anything {target_res} m Obs vs Pred")
        lim_min = min(y_test.min(), y_pred_test.min())
        lim_max = max(y_test.max(), y_pred_test.max())
        axs[0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--')

        residuals = y_test - y_pred_test
        axs[1].hist(residuals, bins=30, edgecolor='black')
        axs[1].set_xlabel("Residuals Test")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Residual Distribution")

        axs[2].barh(importance_df["Covariate"], importance_df["Importance"])
        axs[2].invert_yaxis()
        axs[2].set_xlabel("Importance")
        axs[2].set_title("Feature Importance")

        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        plt.close()

        feedback.pushInfo(f"Train R2={r2_tr:.4f} RMSE={rmse_tr:.4f} MAE={mae_tr:.4f}")
        feedback.pushInfo(f"Test  R2={r2_te:.4f} RMSE={rmse_te:.4f} MAE={mae_te:.4f}")
        if oob is not None:
            feedback.pushInfo(f"OOB R2 about {oob:.4f}")

        # Cleanup temps
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                feedback.pushInfo(f"Failed to remove temp {p} because {e}")

        return {
            self.OUTPUT_RASTER: output_raster,
            self.OUTPUT_CSV: output_csv,
            self.OUTPUT_PLOT: output_plot
        }
