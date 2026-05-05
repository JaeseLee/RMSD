import os
import glob as g
import numpy as np
import pandas as pd
import rasterio as rio
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling


def parse_s1_date(path):
    """
    예:
    S1A_IW_GRDH_1SDV_20190211T194432_20190211T194457_...
    """
    name = os.path.basename(path)
    return datetime.strptime(name.split('_')[5].split('T')[0], '%Y%m%d')


def parse_s2_date(path):
    """
    예:
    S2_2019-02-13_B04.tif
    """
    name = os.path.basename(path)
    return datetime.strptime(name.split('_')[1], '%Y-%m-%d')

def read_tif(path):
    with rio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr <= -9990] = np.nan

    return arr


def read_reproject_to_match(src_path, match_path, resampling=Resampling.bilinear):
    """
    src_path raster를 match_path raster의 CRS, transform, shape에 맞춰 읽음.
    """
    with rio.open(match_path) as match:
        dst_crs = match.crs
        dst_transform = match.transform
        dst_height = match.height
        dst_width = match.width

    with rio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)

        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr[src_arr == src_nodata] = np.nan
        src_arr[src_arr <= -9990] = np.nan

        dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling
        )

    return dst_arr

def estimate_cloud_fraction(
    R,
    G,
    B,
    Nir,
    brightness_threshold=0.28,
    whiteness_threshold=0.16,
    ndvi_threshold=0.35,
    min_valid_fraction=0.7,
):
    """
    Cloud-mask band가 없을 때 S2 optical reflectance로 scene-level cloud 가능성을 추정.
    입력 band는 0-1 scale을 가정한다.

    구름 후보 조건:
    - visible RGB 평균이 밝음
    - R/G/B 차이가 작아 흰색에 가까움
    - NDVI가 낮음
    """
    valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B) & np.isfinite(Nir)
    valid_fraction = np.mean(valid)

    if valid_fraction < min_valid_fraction:
        return np.nan, valid_fraction

    visible_mean = (R + G + B) / 3.0
    visible_range = np.maximum.reduce([R, G, B]) - np.minimum.reduce([R, G, B])
    ndvi = (Nir - R) / (Nir + R)

    cloud_like = (
        valid
        & (visible_mean > brightness_threshold)
        & (visible_range < whiteness_threshold)
        & (ndvi < ndvi_threshold)
    )

    return np.sum(cloud_like) / np.sum(valid), valid_fraction

def build_matched_df_one_folder(folder):
    """
    한 scene folder 안에서 동일 날짜의 S1 VV/VH와 S2 B02/B03/B04/B08/B11/B12를 매칭.
    """

    VVs = sorted(g.glob(os.path.join(folder, '*VV*.tif')))
    VHs = sorted(g.glob(os.path.join(folder, '*VH*.tif')))

    Rs = sorted(g.glob(os.path.join(folder, '*B04*.tif')))
    Gs = sorted(g.glob(os.path.join(folder, '*B03*.tif')))
    Bs = sorted(g.glob(os.path.join(folder, '*B02*.tif')))
    Nirs = sorted(g.glob(os.path.join(folder, '*B08*.tif')))
    B11s = sorted(g.glob(os.path.join(folder, '*B11*.tif')))
    B12s = sorted(g.glob(os.path.join(folder, '*B12*.tif')))

    # -----------------------------
    # S1 dataframe
    # -----------------------------
    df_vv = pd.DataFrame({
        'VV': VVs,
        'date': [parse_s1_date(f) for f in VVs]
    })

    df_vh = pd.DataFrame({
        'VH': VHs,
        'date': [parse_s1_date(f) for f in VHs]
    })

    df_s1 = df_vv.merge(df_vh, on='date', how='inner')

    # -----------------------------
    # S2 dataframe
    # -----------------------------
    df_r = pd.DataFrame({
        'R': Rs,
        'date': [parse_s2_date(f) for f in Rs]
    })

    df_g = pd.DataFrame({
        'G': Gs,
        'date': [parse_s2_date(f) for f in Gs]
    })

    df_b = pd.DataFrame({
        'B': Bs,
        'date': [parse_s2_date(f) for f in Bs]
    })

    df_nir = pd.DataFrame({
        'Nir': Nirs,
        'date': [parse_s2_date(f) for f in Nirs]
    })

    df_b11 = pd.DataFrame({
        'B11': B11s,
        'date': [parse_s2_date(f) for f in B11s]
    })

    df_b12 = pd.DataFrame({
        'B12': B12s,
        'date': [parse_s2_date(f) for f in B12s]
    })

    df_s2 = (
        df_r
        .merge(df_g, on='date', how='inner')
        .merge(df_b, on='date', how='inner')
        .merge(df_nir, on='date', how='inner')
        .merge(df_b11, on='date', how='inner')
        .merge(df_b12, on='date', how='inner')
    )

    # -----------------------------
    # 같은 날짜만 매칭
    # -----------------------------
    df_all = df_s1.merge(df_s2, on='date', how='inner')
    df_all['folder'] = folder

    return df_all.sort_values('date').reset_index(drop=True)

SEN12FLOOD_root = "/Users/jslee/Downloads/SEN12FLOOD"

trns = sorted([
    f for f in g.glob(os.path.join(SEN12FLOOD_root, "trn", "*"))
    if os.path.isdir(f)
])

tsts = sorted([
    f for f in g.glob(os.path.join(SEN12FLOOD_root, "tst", "*"))
    if os.path.isdir(f)
])

print("train folders:", len(trns))
print("test folders :", len(tsts))

def build_matched_df_from_folders(folders):
    df_list = []

    for folder in tqdm(folders, desc="building matched dataframe"):
        try:
            df_tmp = build_matched_df_one_folder(folder)

            if len(df_tmp) > 0:
                df_list.append(df_tmp)

        except Exception as e:
            print(folder, "failed:", e)

    if len(df_list) == 0:
        return pd.DataFrame()

    df_matched = pd.concat(df_list, ignore_index=True)
    return df_matched

def make_xy_from_matched_df(
    df_matched,
    expected_shape=(512, 512),
    sar_resampling=Resampling.bilinear,
    optical_scale=10000.0,
    convert_sar_db_to_linear_for_rvi=True,
    skip_cloudy_scene=True,
    max_cloud_fraction=0.01,
    cloud_brightness_threshold=0.28,
    cloud_whiteness_threshold=0.16,
    cloud_ndvi_threshold=0.35,
    min_optical_valid_fraction=0.7,
):
    """
    X: [VV, VH, RVI]
    Y: [NDVI, MNDWI, NBR]

    Returns
    -------
    X_arr : np.ndarray
        shape = [N, 3, 512, 512]
    Y_arr : np.ndarray
        shape = [N, 3, 512, 512]
    df_good : pd.DataFrame
        successfully processed rows
    df_bad : pd.DataFrame
        failed/skipped rows with reason
    """

    X_list = []
    Y_list = []
    good_records = []
    bad_rows = []

    for i in tqdm(range(len(df_matched)), desc="making X/Y"):
        row = df_matched.iloc[i]

        try:
            # S2 R band grid를 기준으로 SAR를 맞춤
            match_path = row["R"]

            VV = read_reproject_to_match(
                row["VV"],
                match_path,
                resampling=sar_resampling
            )

            VH = read_reproject_to_match(
                row["VH"],
                match_path,
                resampling=sar_resampling
            )

            VV[VV<0]=np.nan
            VH[VH<0]=np.nan

            B = read_reproject_to_match(row["B"], match_path, resampling=Resampling.bilinear) / optical_scale
            G = read_reproject_to_match(row["G"], match_path, resampling=Resampling.bilinear) / optical_scale
            R = read_tif(row["R"]) / optical_scale
            Nir = read_reproject_to_match(row["Nir"], match_path, resampling=Resampling.bilinear) / optical_scale
            B11 = read_reproject_to_match(row["B11"], match_path, resampling=Resampling.bilinear) / optical_scale
            B12 = read_reproject_to_match(row["B12"], match_path, resampling=Resampling.bilinear) / optical_scale

            # shape 체크
            shapes = [VV.shape, VH.shape, B.shape, G.shape, R.shape, Nir.shape, B11.shape, B12.shape]

            if len(set(shapes)) != 1:
                bad_rows.append((i, row["folder"], row["date"], "shape mismatch", shapes))
                continue

            if expected_shape is not None and VV.shape != expected_shape:
                bad_rows.append((i, row["folder"], row["date"], "not expected shape", shapes))
                continue

            cloud_fraction, optical_valid_fraction = estimate_cloud_fraction(
                R,
                G,
                B,
                Nir,
                brightness_threshold=cloud_brightness_threshold,
                whiteness_threshold=cloud_whiteness_threshold,
                ndvi_threshold=cloud_ndvi_threshold,
                min_valid_fraction=min_optical_valid_fraction,
            )

            if not np.isfinite(cloud_fraction):
                bad_rows.append((
                    i,
                    row["folder"],
                    row["date"],
                    "low optical valid fraction",
                    {
                        "optical_valid_fraction": optical_valid_fraction,
                        "min_optical_valid_fraction": min_optical_valid_fraction,
                    }
                ))
                continue

            if skip_cloudy_scene and cloud_fraction > max_cloud_fraction:
                bad_rows.append((
                    i,
                    row["folder"],
                    row["date"],
                    "cloudy optical scene",
                    {
                        "cloud_fraction": cloud_fraction,
                        "max_cloud_fraction": max_cloud_fraction,
                        "optical_valid_fraction": optical_valid_fraction,
                    }
                ))
                continue

            # -------------------------------------------------
            # RVI 계산
            # -------------------------------------------------
            # S1 자료가 dB일 가능성이 있으면 linear로 변환해서 RVI 계산
            if convert_sar_db_to_linear_for_rvi:
                if np.nanmedian(VV) < 0 and np.nanmedian(VH) < 0:
                    VV_for_rvi = 10 ** (VV / 10.0)
                    VH_for_rvi = 10 ** (VH / 10.0)
                else:
                    VV_for_rvi = VV
                    VH_for_rvi = VH
            else:
                VV_for_rvi = VV
                VH_for_rvi = VH

            RVI = (4.0 * VH_for_rvi) / (VV_for_rvi + VH_for_rvi)

            # Optical indices 계산
            NDVI = (Nir - R) / (Nir + R)
            MNDWI = (G - B11) / (G + B11)
            NBR = (Nir - B12) / (Nir + B12)

            RVI[~np.isfinite(RVI)] = np.nan
            NDVI[~np.isfinite(NDVI)] = np.nan
            MNDWI[~np.isfinite(MNDWI)] = np.nan
            NBR[~np.isfinite(NBR)] = np.nan

            X = np.stack([VV, VH, RVI], axis=0).astype(np.float32)
            Y = np.stack([NDVI, MNDWI, NBR], axis=0).astype(np.float32)

            X_list.append(X)
            Y_list.append(Y)
            good_records.append({
                "idx": i,
                "cloud_fraction": cloud_fraction,
                "optical_valid_fraction": optical_valid_fraction,
            })

        except Exception as e:
            bad_rows.append((i, row.get("folder", None), row.get("date", None), str(e), None))

    if len(X_list) == 0:
        X_arr = np.empty((0, 3, expected_shape[0], expected_shape[1]), dtype=np.float32)
        Y_arr = np.empty((0, 3, expected_shape[0], expected_shape[1]), dtype=np.float32)
    else:
        X_arr = np.stack(X_list, axis=0).astype(np.float32)
        Y_arr = np.stack(Y_list, axis=0).astype(np.float32)

    if len(good_records) == 0:
        df_good = df_matched.iloc[[]].reset_index(drop=True)
        df_good["cloud_fraction"] = []
        df_good["optical_valid_fraction"] = []
    else:
        good_info = pd.DataFrame(good_records)
        df_good = df_matched.iloc[good_info["idx"].to_numpy()].reset_index(drop=True)
        df_good["cloud_fraction"] = good_info["cloud_fraction"].to_numpy()
        df_good["optical_valid_fraction"] = good_info["optical_valid_fraction"].to_numpy()

    df_bad = pd.DataFrame(
        bad_rows,
        columns=["idx", "folder", "date", "reason", "detail"]
    )

    return X_arr, Y_arr, df_good, df_bad

def process_split(
    split_name,
    folders,
    out_dir,
    expected_shape=(512, 512),
    skip_cloudy_scene=True,
    max_cloud_fraction=0.01,
    cloud_brightness_threshold=0.28,
    cloud_whiteness_threshold=0.16,
    cloud_ndvi_threshold=0.35,
    min_optical_valid_fraction=0.7,
):
    """
    split_name: 'trn' or 'tst'
    folders: train/test folder list
    out_dir: output directory
    """

    print("=" * 70)
    print(f"Processing split: {split_name}")
    print(f"N folders: {len(folders)}")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # 1. same-date S1/S2 matched dataframe
    # -------------------------------------------------
    df_matched = build_matched_df_from_folders(folders)

    print(f"{split_name} matched dataframe:", df_matched.shape)

    if len(df_matched) == 0:
        print(f"No matched samples for {split_name}")
        return None, None, df_matched, None, None

    df_matched.to_csv(
        os.path.join(out_dir, f"{split_name}_matched_samples_all.csv"),
        index=False
    )

    # -------------------------------------------------
    # 2. make X/Y arrays
    # -------------------------------------------------
    X_arr, Y_arr, df_good, df_bad = make_xy_from_matched_df(
        df_matched,
        expected_shape=expected_shape,
        sar_resampling=Resampling.bilinear,
        optical_scale=10000.0,
        convert_sar_db_to_linear_for_rvi=True,
        skip_cloudy_scene=skip_cloudy_scene,
        max_cloud_fraction=max_cloud_fraction,
        cloud_brightness_threshold=cloud_brightness_threshold,
        cloud_whiteness_threshold=cloud_whiteness_threshold,
        cloud_ndvi_threshold=cloud_ndvi_threshold,
        min_optical_valid_fraction=min_optical_valid_fraction,
    )

    print(f"{split_name} X:", X_arr.shape)
    print(f"{split_name} Y:", Y_arr.shape)
    print(f"{split_name} good samples:", len(df_good))
    print(f"{split_name} bad samples :", len(df_bad))
    if len(df_bad) > 0:
        print(f"{split_name} skipped reasons:")
        print(df_bad["reason"].value_counts())

    # -------------------------------------------------
    # 3. save
    # -------------------------------------------------
    np.save(
        os.path.join(out_dir, f"{split_name}_X_VV_VH_RVI_raw.npy"),
        X_arr
    )

    np.save(
        os.path.join(out_dir, f"{split_name}_Y_NDVI_MNDWI_NBR_raw.npy"),
        Y_arr
    )

    df_good.to_csv(
        os.path.join(out_dir, f"{split_name}_matched_samples_good.csv"),
        index=False
    )

    df_bad.to_csv(
        os.path.join(out_dir, f"{split_name}_matched_samples_bad.csv"),
        index=False
    )

    return X_arr, Y_arr, df_matched, df_good, df_bad

# ==============================================================================
# This code trying to make data to be compatible to U-Net -> (Batch, Channel, Height, Width) shape
# ==============================================================================

out_dir = "/Users/jslee/Downloads/SEN12FLOOD/processed"
os.makedirs(out_dir, exist_ok=True)

X_trn, Y_trn, df_matched_trn, df_good_trn, df_bad_trn = process_split(
    split_name="trn",
    folders=trns,
    out_dir=out_dir,
    expected_shape=(512, 512),
    skip_cloudy_scene=True,
    max_cloud_fraction=0.01
)

X_tst, Y_tst, df_matched_tst, df_good_tst, df_bad_tst = process_split(
    split_name="tst",
    folders=tsts,
    out_dir=out_dir,
    expected_shape=(512, 512),
    skip_cloudy_scene=True,
    max_cloud_fraction=0.01
)
