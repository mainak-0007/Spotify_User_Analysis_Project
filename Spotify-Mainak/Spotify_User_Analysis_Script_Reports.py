#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Spotify_User_Analysis_Final_Script.ipynb
Conversion Date: 2025-10-29T17:57:06.666Z
"""

# ==============================================================
# 📦 INSTALL & IMPORT ALL REQUIRED LIBRARIES
# ==============================================================

# --- Install missing libraries (safe even if already installed)
# The %pip magic works directly inside Jupyter Notebook
%pip install pandas numpy matplotlib seaborn scikit-learn scipy

# --- Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Optional: Style preferences for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

print("✅ All required libraries imported successfully!")


from pathlib import Path  # <-- Add this line

# Define your base folder (Downloads in this case)
base_dir = Path(r"C:\Users\ritaj\Downloads")

# Create a new subfolder for all analysis outputs
project_folder = base_dir / "Spotify_User_Analysis"
project_folder.mkdir(parents=True, exist_ok=True)  # creates folder if it doesn't exist

# Input CSV path
INPUT_PATH = base_dir / "user_song_merged_exact.csv"   # adjust if the CSV is elsewhere

# Output CSV name (cleaned and merged file)
OUTPUT_NAME = "updated_user_song_merged_exact.csv"
OUTPUT_PATH = project_folder / OUTPUT_NAME

print("✅ Input CSV:", INPUT_PATH)
print("📂 Output folder for all analysis:", project_folder)
print("💾 Cleaned CSV will be saved to:", OUTPUT_PATH)


# Cell 2: load dataset and inspect
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"File not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
print("Original dataframe shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head(6))

# Cell 3: duplicates diagnostics
if 'track_id' not in df.columns:
    raise KeyError("Column 'track_id' not found in the dataset. Please confirm the column name.")

total_exact_dupes = df.duplicated(keep=False).sum()
rows_sharing_track_id = df.duplicated(subset=['track_id'], keep=False).sum()
unique_track_ids = df['track_id'].nunique()

print(f"Exact duplicate rows (full-row duplicates): {total_exact_dupes}")
print(f"Rows that share a track_id with at least one other row: {rows_sharing_track_id}")
print(f"Unique track_id count: {unique_track_ids}")

import pandas as pd
import numpy as np
import re
from collections import Counter

# Cell 3: merge by track_id, sum play_count, cleaned track_name, average numeric features, save result.

def clean_title(title: str) -> str:
    """Remove bracketed content and normalize whitespace."""
    if pd.isna(title):
        return ""
    s = str(title)
    s = re.sub(r"\s*[\(\[\{].*?[\)\]\}]\s*", " ", s)   # remove ( ... ) [ ... ] { ... }
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Ensure play_count is present and numeric
if 'play_count' not in df.columns:
    raise KeyError("Column 'play_count' is required but not found in the dataset.")
df['play_count'] = pd.to_numeric(df['play_count'], errors='coerce').fillna(0).astype(int)

# Candidate numeric features to average
numeric_candidates = [
    'danceability','energy','acousticness','valence','speechiness',
    'instrumentalness','liveness','tempo','loudness','duration_ms','popularity'
]
numeric_features = [c for c in numeric_candidates if c in df.columns]

# Metadata columns to keep
meta_candidates = ['artist','album_name','track_genre','artist_clean','track_name_clean']
meta_cols = [c for c in meta_candidates if c in df.columns]

print("Numeric features (averaged):", numeric_features)
print("Meta columns (first non-null kept):", meta_cols)

records = []
grouped = df.groupby('track_id', sort=False)

for tid, g in grouped:
    rec = {'track_id': tid}
    rec['play_count'] = int(g['play_count'].sum())

    # average numeric features
    for col in numeric_features:
        rec[col] = float(g[col].mean(skipna=True)) if col in g.columns else np.nan

    # create canonical cleaned track_name
    if 'track_name' in g.columns:
        originals = g['track_name'].astype(str).fillna("").tolist()
        cleaned_list = [clean_title(t) for t in originals if t and str(t).lower() != 'nan']
        cleaned_list = [c for c in cleaned_list if c]
        if cleaned_list:
            canonical = Counter(cleaned_list).most_common(1)[0][0]
        else:
            first_nonnull = g['track_name'].dropna().astype(str)
            canonical = clean_title(first_nonnull.iloc[0]) if not first_nonnull.empty else ""
    else:
        canonical = ""
    rec['track_name'] = canonical

    # keep first non-null for other meta columns
    for m in meta_cols:
        if m in g.columns:
            nonnull = g[m].dropna()
            rec[m] = nonnull.iloc[0] if not nonnull.empty else np.nan
        else:
            rec[m] = np.nan

    records.append(rec)

df_agg = pd.DataFrame.from_records(records)

# Reorder columns
ordered = ['track_id', 'track_name', 'play_count'] + numeric_features + meta_cols
ordered = [c for c in ordered if c in df_agg.columns]
df_agg = df_agg[ordered]

# Save the updated CSV
df_agg.to_csv(OUTPUT_PATH, index=False)
print("✅ Saved aggregated cleaned file to:", OUTPUT_PATH)


# Cell 4: verify results and preview merged rows
print("Original rows:", len(df))
print("Rows after aggregation (unique track_id):", len(df_agg))
print("Reduction (rows merged):", len(df) - len(df_agg))

# show a sample of merged track_ids that had >1 rows originally
dupe_counts = df['track_id'].value_counts()
duped_ids = dupe_counts[dupe_counts > 1].index.tolist()[:10]  # up to 10 examples
print("Example track_ids merged (up to 10):", duped_ids)

if duped_ids:
    display(df_agg[df_agg['track_id'].isin(duped_ids)].head(10))

# show top 10 by play_count in the aggregated file
display(df_agg.sort_values('play_count', ascending=False).head(10))


# Cell: verify total play count across the dataset
total_play_count = df_agg['play_count'].sum()
print(f"🎵 Total play count across all unique songs (after merging): {total_play_count:,}")


before_total = df['play_count'].sum()
after_total = df_agg['play_count'].sum()

print(f"Total play_count before merging: {before_total:,}")
print(f"Total play_count after merging : {after_total:,}")

if before_total == after_total:
    print("✅ Perfect — totals match! (No loss during merge)")
else:
    print(f"⚠️ Totals differ by {after_total - before_total:,} — check for NaN conversions or duplicates.")


missing_ids = df[df['track_id'].isna()]
print("Rows with missing track_id:", len(missing_ids))


print("Total rows:", len(df))
print("Unique track_ids:", df['track_id'].nunique())


before_total = df['play_count'].sum()
after_total = df_agg['play_count'].sum()
difference = after_total - before_total

print(f"Before merging: {before_total:,}")
print(f"After merging : {after_total:,}")
print(f"Difference    : {difference:,}")

# Check missing track IDs or play_count anomalies
print("\nMissing track_id rows:", df['track_id'].isna().sum())
print("Rows with non-numeric play_count originally:",
      (pd.to_numeric(df['play_count'], errors='coerce').isna()).sum())


df_agg.to_csv(OUTPUT_PATH, index=False)
print("Saved aggregated cleaned file to:", OUTPUT_PATH)

# Select numeric audio features
feature_candidates = [
    'danceability','energy','acousticness','valence','speechiness',
    'instrumentalness','liveness','tempo','loudness','duration_ms','popularity'
]
features = [c for c in feature_candidates if c in df.columns]
target = 'play_count'

# Drop missing and compute log play_count
data = df[features + [target]].dropna().copy()
data['play_count_log'] = np.log1p(data['play_count'])

print(f"Rows used for analysis: {len(data)}")
print("Features:", features)


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr, spearmanr
# from pathlib import Path

# # compute correlations
# pearson_results = {}
# spearman_results = {}
# pearson_p = {}
# spearman_p = {}

# for f in features:
#     x = data[f].values
#     y = data['play_count_log'].values

#     try:
#         r, p = pearsonr(x, y)
#     except Exception as e:
#         r, p = np.nan, np.nan
#         print(f"Pearson failed for {f}: {e}")

#     try:
#         sr, sp = spearmanr(x, y)
#     except Exception as e:
#         sr, sp = np.nan, np.nan
#         print(f"Spearman failed for {f}: {e}")

#     pearson_results[f] = r
#     pearson_p[f] = p
#     spearman_results[f] = sr
#     spearman_p[f] = sp

# pearson_df = pd.DataFrame({
#     'feature': list(pearson_results.keys()),
#     'pearson_r': list(pearson_results.values()),
#     'pearson_p': [pearson_p[k] for k in pearson_results.keys()]
# }).set_index('feature').sort_values('pearson_r', ascending=False)

# spearman_df = pd.DataFrame({
#     'feature': list(spearman_results.keys()),
#     'spearman_r': list(spearman_results.values()),
#     'spearman_p': [spearman_p[k] for k in spearman_results.keys()]
# }).set_index('feature').sort_values('spearman_r', ascending=False)

# print("Pearson correlations (top):")
# display(pearson_df)
# print("\nSpearman correlations (top):")
# display(spearman_df)

# # combined bar plot (absolute magnitude)
# comp = pd.concat([pearson_df['pearson_r'].abs(), spearman_df['spearman_r'].abs()], axis=1)
# comp.columns = ['abs_pearson_r', 'abs_spearman_r']
# comp = comp.fillna(0).sort_values(by='abs_pearson_r', ascending=False)

# # --- Spotify-styled dark plot ---
# spotify_green = "#1DB954"
# background_color = "black"
# text_color = "white"

# plt.style.use("default")
# fig, ax = plt.subplots(figsize=(10, 6))
# fig.patch.set_facecolor(background_color)
# ax.set_facecolor(background_color)

# comp.plot(kind='bar', rot=45, ax=ax, color=[spotify_green, spotify_green])

# # Set title and labels
# ax.set_title("Absolute Pearson vs Spearman correlation with log(play_count)", color=text_color, fontsize=14, pad=15)
# ax.set_ylabel("Absolute correlation coefficient", color=text_color, fontsize=12)
# ax.set_xlabel("Feature", color=text_color, fontsize=12)

# # Change tick and spine colors
# ax.tick_params(colors=text_color, labelcolor=text_color)
# for spine in ax.spines.values():
#     spine.set_color(text_color)

# # Make legend white text
# legend = ax.legend(facecolor=background_color, edgecolor=text_color, labelcolor=text_color)
# for text in legend.get_texts():
#     text.set_color(text_color)

# # Optional: add values above bars (white text)
# for container in ax.containers:
#     ax.bar_label(container, fmt="%.2f", label_type='edge', color=text_color, fontsize=9)

# plt.tight_layout()

# # Save plot
# project_folder = Path(r"C:\Users\ritaj\Downloads\Spotify_User_Analysis")
# project_folder.mkdir(parents=True, exist_ok=True)
# out_file = project_folder / "pearson_spearman_comparison.png"
# plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=background_color)
# print("✅ Saved plot to:", out_file)

# plt.show()

print("✅ Code is commented out successfully, and this cell still runs!")


# corr_matrix = data[features + ['play_count_log']].corr(method='pearson')
# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap (Pearson)")
# plt.tight_layout()
# plt.show()

print("✅ Code is commented out successfully, and this cell still runs!")



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr, spearmanr
# from pathlib import Path

# # Example: Assuming you already computed your features, data, and correlations as before
# # Replace this block with your real 'data', 'features', and correlation computations.

# # --- Spotify-styled scatter plot with regression line ---
# spotify_green = "#1DB954"
# background_color = "black"
# text_color = "white"

# plt.style.use("default")

# # Example feature plot: change `feature_name` to the feature you want to visualize
# feature_name = features[0]  # for example
# x = data[feature_name]
# y = data["play_count_log"]

# fig, ax = plt.subplots(figsize=(8, 6))

# # Set dark background
# fig.patch.set_facecolor(background_color)
# ax.set_facecolor(background_color)

# # Plot scatter: dots in alternating white and gray for texture
# colors = np.where(np.arange(len(x)) % 2 == 0, "white", "gray")
# ax.scatter(x, y, color=colors, alpha=0.7, edgecolor="none", s=40)

# # Add regression line in Spotify green
# sns.regplot(
#     x=x,
#     y=y,
#     scatter=False,
#     color=spotify_green,
#     line_kws={"lw": 2},
#     ax=ax
# )

# # Title and labels in white
# ax.set_title(f"Correlation between {feature_name} and log(play_count)", color=text_color, fontsize=14, pad=15)
# ax.set_xlabel(feature_name, color=text_color, fontsize=12)
# ax.set_ylabel("log(play_count)", color=text_color, fontsize=12)

# # White ticks and spines
# ax.tick_params(colors=text_color, labelcolor=text_color)
# for spine in ax.spines.values():
#     spine.set_color(text_color)

# # Optional: annotate correlation values (white text)
# r, _ = pearsonr(x, y)
# ax.text(0.05, 0.95, f"Pearson r = {r:.2f}", color=text_color, transform=ax.transAxes,
#         fontsize=12, ha="left", va="top")

# plt.tight_layout()

# # Save Spotify-styled plot
# project_folder = Path(r"C:\Users\ritaj\Downloads\Spotify_User_Analysis")
# project_folder.mkdir(parents=True, exist_ok=True)
# out_file = project_folder / f"{feature_name}_spotify_theme_correlation.png"
# plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=background_color)
# print("✅ Saved Spotify-themed correlation plot to:", out_file)

# # plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Prepare data for regression (adjust your dataframe name if needed)
X = data[features].values
y = data['play_count_log'].values

# Scale features (optional but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test) 


# --- Full plotting & saving block (Spotify dark theme) ---
# Assumes the following variables are already defined in your notebook:
#   linreg, features, X_test, X_train, y_test, y_pred_lr, y_pred_lr (predictions),
#   and that pandas as pd, numpy as np, matplotlib.pyplot as plt are available.
# This block is compatible with older scikit-learn (computes RMSE via sqrt).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
from sklearn.metrics import r2_score, mean_squared_error

# --- Save location (Ritaj folder) ---
project_folder = Path(r"C:\Users\ritaj\Downloads\Spotify_User_Analysis")
project_folder.mkdir(parents=True, exist_ok=True)

# --- Theme colors ---
SPOTIFY_GREEN = "#1DB954"
BACKGROUND_COLOR = "black"
TEXT_COLOR = "white"
WHITE = "white"
GRAY = "gray"

# --- Helper to apply Spotify-style formatting to axes ---
def style_axes(ax):
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

# --- Compute metrics (RMSE compatible with older sklearn) ---
r2 = r2_score(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))  # use sqrt for compatibility
print(f"Linear Regression R²: {r2:.4f}")
print(f"Linear Regression RMSE: {rmse:.4f}")

# --- Save coefficients to CSV ---
coeffs = pd.Series(linreg.coef_, index=features).sort_values(key=lambda s: s.abs(), ascending=False)
coeffs_df = coeffs.reset_index().rename(columns={'index': 'feature', 0: 'coefficient'}).assign(abs_coeff=coeffs.abs().values)
coeffs_csv = project_folder / "linear_regression_coefficients.csv"
coeffs_df.to_csv(coeffs_csv, index=False)
print("Saved coefficients to:", coeffs_csv)
display(coeffs_df)

# --- 1) Predicted vs Actual scatter (black bg, white/gray dots, green y=x line) ---
plt.style.use("default")
fig, ax = plt.subplots(figsize=(7,6))
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

# alternating white/gray points for texture
colors = np.where(np.arange(len(y_test)) % 2 == 0, WHITE, GRAY)
ax.scatter(y_test, y_pred_lr, alpha=0.85, s=40, c=colors, edgecolor="none")

# diagonal y=x line -> Spotify green
minv = min(np.nanmin(y_test), np.nanmin(y_pred_lr))
maxv = max(np.nanmax(y_test), np.nanmax(y_pred_lr))
ax.plot([minv, maxv], [minv, maxv], color=SPOTIFY_GREEN, linestyle='--', linewidth=1)

ax.set_xlabel("Actual log(1+play_count)")
ax.set_ylabel("Predicted log(1+play_count)")
ax.set_title("Predicted vs Actual (Linear Regression)")
style_axes(ax)

plt.tight_layout()
pvafile = project_folder / "lr_predicted_vs_actual.png"
plt.savefig(pvafile, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
print("Saved:", pvafile)
plt.show()

# --- 2) Residuals vs Predicted scatter (black bg, white/gray dots, green zero-line) ---
residuals = y_test - y_pred_lr
fig, ax = plt.subplots(figsize=(7,6))
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

colors = np.where(np.arange(len(y_pred_lr)) % 2 == 0, WHITE, GRAY)
ax.scatter(y_pred_lr, residuals, alpha=0.85, s=40, c=colors, edgecolor="none")

# zero horizontal line -> Spotify green
ax.axhline(0, color=SPOTIFY_GREEN, linestyle='--', linewidth=1)

ax.set_xlabel("Predicted log(1+play_count)")
ax.set_ylabel("Residuals (Actual - Predicted)")
ax.set_title("Residuals vs Predicted")
style_axes(ax)

plt.tight_layout()
resfile = project_folder / "lr_residuals_vs_predicted.png"
plt.savefig(resfile, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
print("Saved:", resfile)
plt.show()

# --- 3a) Coefficient bar chart (signed) ---
fig, ax = plt.subplots(figsize=(8,4))
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

signed_series = coeffs.sort_values(ascending=True)  # show negative on top if present
signed_series.plot(kind='barh', ax=ax, color=SPOTIFY_GREEN, edgecolor="none")

ax.set_xlabel("Coefficient value")
ax.set_title("Linear Regression Coefficients (signed)")
style_axes(ax)

plt.tight_layout()
coef_signed_file = project_folder / "lr_coefficients_signed.png"
plt.savefig(coef_signed_file, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
print("Saved:", coef_signed_file)
plt.show()

# --- 3b) Coefficient bar chart (absolute importance) ---
fig, ax = plt.subplots(figsize=(8,4))
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

abs_series = coeffs.abs().sort_values(ascending=True)
abs_series.plot(kind='barh', ax=ax, color=SPOTIFY_GREEN, edgecolor="none")

ax.set_xlabel("Absolute coefficient value")
ax.set_title("Linear Regression Coefficients (absolute importance)")
style_axes(ax)

plt.tight_layout()
coef_abs_file = project_folder / "lr_coefficients_abs.png"
plt.savefig(coef_abs_file, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
print("Saved:", coef_abs_file)
plt.show()

# --- Save predictions/residuals CSV ---
preds_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_lr,
    'residual': residuals
})
preds_csv = project_folder / "lr_predictions_and_residuals.csv"
preds_df.to_csv(preds_csv, index=False)
print("Saved predictions/residuals to:", preds_csv)


# --- FULL RANDOM FOREST + SPOTIFY-THEMED PLOTS (robust, loads df_agg if needed) ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# ========== THEME & SETTINGS ==========
spotify_green = "#1DB954"
background_color = "black"
text_color = "white"
dot_gray = "#8a8a8a"

plt.style.use("default")

# ---------- Ensure df_agg is available ----------
# If df_agg is not defined in the notebook session, try to load it from the default path.
project_folder = Path(r"C:\Users\ritaj\Downloads\Spotify_User_Analysis")
csv_path = project_folder / "updated_user_song_merged_exact.csv"

if 'df_agg' not in globals():
    if csv_path.exists():
        print(f"Loading data from: {csv_path}")
        df_agg = pd.read_csv(csv_path)
    else:
        raise NameError(
            "df_agg is not defined and the default CSV was not found.\n"
            f"Tried: {csv_path}\n"
            "Please either define `df_agg` in a previous cell or place the CSV at that path."
        )
else:
    print("Using existing df_agg in memory.")

# ---------- Feature list and target ----------
# Provided example features; we'll keep only those that exist in df_agg
requested_features = [
    "danceability","energy","acousticness","valence","speechiness",
    "instrumentalness","liveness","tempo","loudness","duration_ms","popularity"
]
target = "play_count"

# Validate features exist in df_agg
existing_features = [f for f in requested_features if f in df_agg.columns]
missing = [f for f in requested_features if f not in df_agg.columns]
if missing:
    print(f"Warning: these requested features were not found in df_agg and will be ignored: {missing}")
if not existing_features:
    raise ValueError("None of the requested features are present in df_agg. Please supply valid numeric feature columns.")

features = existing_features  # use the filtered list
print("Using features:", features)

# Validate target exists
if target not in df_agg.columns:
    raise ValueError(f"Target column '{target}' not found in df_agg. Available columns: {list(df_agg.columns)[:30]}")

# ---------- Drop rows with NaN values in needed columns ----------
df_agg = df_agg.dropna(subset=features + [target]).copy()
if df_agg.shape[0] == 0:
    raise ValueError("After dropping NaNs, no rows remain. Check your data.")

# ---------- Create play class (High/Low) ----------
threshold = df_agg[target].median()
df_agg["play_class"] = np.where(df_agg[target] >= threshold, "High", "Low")
print(f"Play-count threshold (median): {threshold}")

# ---------- Train/Test Split ----------
X = df_agg[features]
y = df_agg[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Train Random Forest ----------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ---------- Evaluate Model ----------
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"✅ Random Forest R²: {r2_rf:.4f}")
print(f"✅ Random Forest RMSE: {rmse_rf:.4f}")

# ---------- 1️⃣ COUNT PLOT (Spotify Dark) ----------
plt.style.use("dark_background")
plt.figure(figsize=(6,4))
custom_palette = {'Low': '#FF3B3B', 'High': spotify_green}
sns.countplot(x='play_class', data=df_agg, hue='play_class', palette=custom_palette, order=['Low', 'High'], legend=False)
plt.title(f"High vs Low Play Count (threshold = {threshold:.0f})", color=text_color, fontsize=13)
plt.xlabel("Play Count Class", color=text_color)
plt.ylabel("Number of Tracks", color=text_color)
plt.tick_params(colors=text_color)
plt.tight_layout()
plt.show()

# ---------- 2️⃣ FEATURE IMPORTANCE (Spotify Theme, horizontal) ----------
importances = rf.feature_importances_
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)
ax.barh(feat_df["Feature"], feat_df["Importance"], color=spotify_green)

# White text styling
ax.set_title("Random Forest Feature Importance", color=text_color, fontsize=14, pad=10)
ax.set_xlabel("Importance", color=text_color, fontsize=12)
ax.set_ylabel("Feature", color=text_color, fontsize=12)
ax.tick_params(colors=text_color)
for spine in ax.spines.values():
    spine.set_color(text_color)

# Annotate each bar with white text
for i, v in enumerate(feat_df["Importance"]):
    ax.text(v + 0.001 * max(feat_df["Importance"].max(), 1), i, f"{v:.3f}", color=text_color, va='center', fontsize=9)

plt.tight_layout()

project_folder.mkdir(parents=True, exist_ok=True)
feat_out = project_folder / "random_forest_feature_importance_spotify.png"
plt.savefig(feat_out, dpi=300, bbox_inches="tight", facecolor=background_color)
print("🎧 Saved Spotify-themed feature importance plot to:", feat_out)
plt.show()

# Spotify-themed Predicted vs Actual plot (final)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# sanity check
required = ['y_test', 'y_pred_rf', 'r2_rf', 'rmse_rf', 'project_folder']
missing = [v for v in required if v not in globals()]
if missing:
    raise NameError(f"Required variable(s) missing from session: {missing}. Run RF training block first.")

# Theme colors
spotify_green = "#1DB954"
background_color = "black"
text_color = "white"
dot_gray = "#8a8a8a"

# Convert to numpy arrays (handle pandas Series)
x = np.asarray(y_test).astype(float)
y = np.asarray(y_pred_rf).astype(float)

# Create figure
fig, ax = plt.subplots(figsize=(7,7))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# Scatter points: alternate white and gray for texture
colors = np.where(np.arange(len(x)) % 2 == 0, "white", dot_gray)
ax.scatter(x, y, color=colors, alpha=0.9, edgecolor='none', s=40, zorder=3)

# Compute and plot linear fit (Spotify green) on top
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = m * x_line + b
ax.plot(x_line, y_line, color=spotify_green, linewidth=2.5, zorder=4, label="Linear fit (Spotify green)")

# 45-degree perfect-prediction line (white dashed)
min_val = min(float(np.nanmin(x)), float(np.nanmin(y)))
max_val = max(float(np.nanmax(x)), float(np.nanmax(y)))
ax.plot([min_val, max_val], [min_val, max_val], color=text_color, linestyle="--", linewidth=1.0, zorder=2, label="Perfect prediction")

# Title, labels, ticks in white
ax.set_title("Random Forest: Predicted vs Actual Play Count", color=text_color, fontsize=14, pad=12)
ax.set_xlabel("Actual Play Count", color=text_color, fontsize=12)
ax.set_ylabel("Predicted Play Count", color=text_color, fontsize=12)
ax.tick_params(colors=text_color, labelsize=10)

# Spines in white
for spine in ax.spines.values():
    spine.set_color(text_color)

# Annotate R² and RMSE in white (top-left)
ax.text(0.03, 0.97, f"R² = {r2_rf:.3f}\nRMSE = {rmse_rf:.3f}",
        transform=ax.transAxes, color=text_color, fontsize=11,
        ha="left", va="top", bbox=dict(facecolor='none', edgecolor='none'))

# Legend with white text
leg = ax.legend(frameon=True, facecolor=background_color, edgecolor=text_color)
for txt in leg.get_texts():
    txt.set_color(text_color)
leg.get_frame().set_edgecolor(text_color)

plt.tight_layout()

# Save (overwrite same filename)
rf_out = Path(project_folder) / "random_forest_actualpredicted_spotify_vs_random_forest_actual.png"
plt.savefig(rf_out, dpi=300, bbox_inches="tight", facecolor=background_color)
print("✅ Saved (overwritten) Spotify-themed Predicted vs Actual plot to:", rf_out)

plt.show()



threshold = df['play_count'].median()
df['play_class'] = np.where(df['play_count'] > threshold, 'High', 'Low')
print(df['play_class'].value_counts())


# --- Class Summary Bar Plot (Dark theme + red/blue classes) ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assuming df_agg and features already exist
# If not, replace df_agg with your main dataframe
# and define features as your numeric column list.

# Compute average feature values by play class
class_summary = df_agg.groupby('play_class')[features + ['play_count']].mean().round(3)
display(class_summary)

# --- Spotify-dark theme + red/blue classes ---
plt.style.use('dark_background')  # black background + white text
plt.figure(figsize=(8,4))

# Define colors for play classes
color_map = {'Low': '#FF3B3B', 'High': '#1E90FF'}  # red & blue shades

# Plot transposed class summary with custom colors
ax = (class_summary[features].T).plot(
    kind='bar',
    color=[color_map[c] for c in class_summary.index],
    edgecolor='white',
    figsize=(8,4)
)

# Titles and labels in white
plt.title("Average Feature Values by Play Class", color='white', fontsize=14)
plt.ylabel("Mean Value", color='white', fontsize=12)
plt.xlabel("Features", color='white', fontsize=12)

# White tick labels
plt.xticks(rotation=45, ha='right', color='white', fontsize=10)
plt.yticks(color='white', fontsize=10)

# White legend with black background
plt.legend(
    title='Play Class',
    facecolor='black',
    edgecolor='white',
    labelcolor='white',
    title_fontsize=10,
    fontsize=10
)

# White spines for neatness
for spine in ax.spines.values():
    spine.set_color('white')

plt.tight_layout()
plt.show()


# Single-cell plotting: class_summary (red/blue) + top_genres & top_artists (Spotify green)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Config ---
plt.style.use('dark_background')   # black background + white text by default
spotify_green = '#1DB954'
color_map = {'Low': '#FF3B3B', 'High': '#1E90FF'}  # red & blue for class_summary

# --- 1) Average Feature Values by Play Class (transposed bar) ---
if 'play_class' in df_agg.columns and len(features) > 0:
    class_summary = df_agg.groupby('play_class')[features + ['play_count']].mean().round(3)
    display(class_summary)

    # Make sure classes are in a deterministic order (Low, High) if present
    classes = [c for c in ['Low', 'High'] if c in class_summary.index]
    if not classes:
        classes = list(class_summary.index)  # fallback to whatever exists

    # Plot: features on x, bars for each class (transposed DataFrame -> features x classes)
    fig, ax = plt.subplots(figsize=(10, 5))
    (class_summary[features].T).plot(
        kind='bar',
        color=[color_map[c] if c in color_map else '#888888' for c in class_summary.index],
        edgecolor='white',
        ax=ax
    )

    # Styling
    ax.set_title("Average Feature Values by Play Class", color='white', fontsize=14)
    ax.set_ylabel("Mean Value", color='white', fontsize=12)
    ax.set_xlabel("Features", color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    ax.legend(title='Play Class', facecolor='black', edgecolor='white', labelcolor='white', title_fontsize=10)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping class_summary plot: 'play_class' column or features list not found in df_agg.")


# --- Helper to style axes consistently ---
def apply_spotify_theme(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color='white', fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, color='white', fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_facecolor('black')   # ensure plotting area is black


# --- 2) Top 10 Genres by Total Play Count (Spotify green) ---
if 'track_genre' in df_agg.columns and 'play_count' in df_agg.columns:
    top_genres = (
        df_agg.groupby('track_genre')['play_count']
              .sum()
              .sort_values(ascending=False)
              .head(10)
              .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_genres, x='track_genre', y='play_count', color=spotify_green, edgecolor='white', ax=ax)

    apply_spotify_theme(ax,
                        title="Top 10 Genres by Total Play Count",
                        xlabel="Genre",
                        ylabel="Total Play Count")
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping genres plot: 'track_genre' or 'play_count' column not found in df_agg.")


# --- 3) Top 10 Artists by Total Play Count (Spotify green) ---
if 'artist' in df_agg.columns and 'play_count' in df_agg.columns:
    top_artists = (
        df_agg.groupby('artist')['play_count']
              .sum()
              .sort_values(ascending=False)
              .head(10)
              .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_artists, x='artist', y='play_count', color=spotify_green, edgecolor='white', ax=ax)

    apply_spotify_theme(ax,
                        title="Top 10 Artists by Total Play Count",
                        xlabel="Artist",
                        ylabel="Total Play Count")
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping artists plot: 'artist' or 'play_count' column not found in df_agg.")