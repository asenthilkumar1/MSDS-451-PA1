# MSDS-451-PA1

MSDS 451 Financial Engineering, Programming Assignment 1.

This project uses machine learning classifiers, including XGBoost gradient boosting, to predict the **direction of the daily return (up or down) of Microsoft (ticker MSFT)** from lagged price, range, and volume features. It applies the course's WTI oil-futures example to Microsoft data, following the same workflow:

1. Retrieve daily price history from Yahoo! Finance.
2. Engineer lagged features with Polars.
3. Select features by all-possible-subsets logistic regression scored with the Akaike Information Criterion (AIC).
4. Tune XGBoost hyperparameters with a randomized search inside time-series cross-validation.
5. Evaluate a final model with a confusion matrix, classification report, and ROC curve.

## Repository contents

| File | Purpose |
|---|---|
| `getdata_yfinance.py` | Downloads daily MSFT price history from Yahoo! Finance and writes it to a CSV. |
| `pa1.py` | Main program: feature engineering, feature selection, tuning, and evaluation. |
| `msds_getdata_yfinance_msft.csv` | MSFT daily data (2000-01-03 to 2025-09-24) produced by `getdata_yfinance.py`; the input to `pa1.py`. |
| `msft-with-computed-features.csv` | Output of `pa1.py`: the data with all computed features, for checking calculations in Excel. |
| `msft_correlation_heatmap.png` | Output of `pa1.py`: correlation heat map of the selected features. |
| `msft_roc_curve.png` | Output of `pa1.py`: ROC curve of the final model. |
| `msft_confusion_matrix.png` | Output of `pa1.py`: confusion matrix of the final model. |

Both `.py` files were exported from Jupyter notebooks, so they contain `# In[n]:` markers and markdown cells as comments. These are ordinary comments; the files run as normal Python scripts, top to bottom.

## Requirements

- Python 3.10 or newer (developed and tested on Python 3.14, macOS, Apple silicon)
- Python packages: `numpy`, `pandas`, `pyarrow`, `polars`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `xgboost`, `yfinance`
- macOS only: the OpenMP runtime, which XGBoost needs (`brew install libomp`)
- Internet access, only for `getdata_yfinance.py`

## Setup

From the repository folder:

```bash
# 1. (macOS only) OpenMP runtime required by XGBoost
brew install libomp

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the packages
pip install numpy pandas pyarrow polars matplotlib seaborn scikit-learn scipy xgboost yfinance
```

Verify the installation (this should print `all imports OK`):

```bash
python3 -c "import numpy, pandas, pyarrow, polars, matplotlib, seaborn, sklearn, scipy, xgboost, yfinance; print('all imports OK')"
```

## Running the program

### Step 1: Get the data (optional)

The repository already contains `msds_getdata_yfinance_msft.csv`, so this step can be skipped. To re-download it:

```bash
python3 getdata_yfinance.py
```

This fetches MSFT history from 2000-01-01 up to (not including) 2025-09-25 and writes `msds_getdata_yfinance_msft.csv`. The ticker, dates, and file prefix are set in the `symbol`, `start_date`, `end_date`, and `startfile` variables near the top of the script. Re-downloading later may give slightly different values than the committed file, because Yahoo! Finance can revise adjusted prices.

### Step 2: Run the analysis

```bash
python3 pa1.py
```

The script reads `msds_getdata_yfinance_msft.csv` from the current directory, so run it from the repository folder. Expect it to take **several minutes**: the feature-selection step fits about 32,000 logistic regressions (every subset of 15 features) and the hyperparameter search runs 100 candidates × 5 folds.

While it runs, it:

- prints the schema, descriptive statistics, and target class counts
- prints the ten lowest-AIC feature subsets
- prints the cross-validation splits and baseline accuracy
- prints the best hyperparameters and cross-validated score
- prints the confusion matrix and classification report
- shows three plots (heat map, ROC curve, confusion matrix) and saves each as a PNG in the working directory
- writes `msft-with-computed-features.csv`

Each `plt.show()` pauses the script until that plot window is closed. To run without any windows (for example on a server), set a non-interactive backend; the PNGs are still saved:

```bash
MPLBACKEND=Agg python3 pa1.py
```

## Testing and verifying the program

The assignment has no separate unit-test suite; the program is verified by running it end to end and checking its outputs.

**1. Syntax check (fast):**

```bash
python3 -c "import ast; [ast.parse(open(f).read()) for f in ('pa1.py', 'getdata_yfinance.py')]; print('OK')"
```

**2. Full run:** `python3 pa1.py` should finish with exit code 0 and no traceback (`echo $?` prints `0`).

**3. Check the outputs.** After a successful run, these should hold:

- `msft-with-computed-features.csv` and the three `msft_*.png` files exist in the folder.
- After the lag rows are dropped, **6,468 rows** remain, with target counts of **3,307 up (1)** and **3,161 down/flat (0)**.
- The ten lowest-AIC subsets have AIC values around 8,947–8,949, and the best subset is features `6 8 10` (`OMCLag1`, `OMCLag3`, `VolumeLag2`).
- Cross-validated accuracy is near **0.50** (baseline) and about **0.52** (tuned), which is only slightly better than a coin flip.
- The final-model confusion matrix sums to 6,468 and the report shows accuracy of about 0.73 (in-sample).

Results use fixed random seeds (`random_state=2025`), so repeated runs on the same data and package versions should give the same numbers; small differences are possible across package versions.

**4. Spot-check the feature engineering** by opening `msft-with-computed-features.csv` in Excel. For example, `CloseLag1` on each row should equal the previous row's `Close`, and `HML` should equal `High - Low`.

## Results summary

| Item | Value |
|---|---|
| Observations used | 6,468 daily rows (after dropping rows with lag nulls) |
| Selected features (lowest AIC) | `OMCLag1`, `OMCLag3`, `VolumeLag2` |
| Cross-validation | `TimeSeriesSplit`, 5 splits, 10-day gap |
| Baseline XGBoost CV accuracy (1,000 trees, defaults) | 0.501 ± 0.021 |
| Best hyperparameters from randomized search | `max_depth=7`, `min_child_weight=2`, `subsample≈0.63`, `learning_rate≈0.01`, `n_estimators=411` |
| Best CV accuracy from search | 0.521 |
| Final model accuracy (in-sample, full data) | 0.73 (ROC AUC 0.72) |

The in-sample figures describe how well the final model fits the data it was trained on. The cross-validated accuracy of about 0.52 is the realistic estimate of out-of-sample performance: next-day direction is only weakly predictable from these lagged features, which is consistent with the near-zero correlations between `LogReturn` and each feature in the heat map.

## Notes on adapting the original WTI example

- All references to WTI were changed to MSFT, including the variable names, CSV names, and narrative text.
- The CSV produced by `yfinance` names a column `Stock Splits` (with a space), so the drop step uses that name.
- The selected features and hyperparameters were re-derived from the MSFT results rather than reused from the WTI example, because the AIC ranking and the search winners depend on the data.
- The notebook-only line `get_ipython().run_line_magic('matplotlib', 'inline')` is commented out so the file runs as a plain script. Uncomment it only when running inside Jupyter.
- `plt.savefig(...)` calls and a `plt.show()` after the ROC curve were added so every plot is both saved and displayed.
- Heat map axes are labeled with feature names.

## Use of AI assistants

I used **Claude Code** (Anthropic's AI coding assistant, running in VS Code) while completing this assignment. It was used as a coding aid; I directed the work and reviewed the results.

What the assistant did:

- **Cleaned up the starter files.** The professor's code arrived as text pasted from a PDF export with broken line wraps and lost indentation. The assistant rejoined wrapped lines, restored comment markers, and fixed indentation in `pa1.py` and `getdata_yfinance.py`.
- **Set up the environment.** It created the virtual environment, installed the Python packages and the OpenMP runtime, and diagnosed an XGBoost import failure.
- **Ran the program and reported results.** It executed `pa1.py` end to end, read the AIC ranking and hyperparameter-search results, and used them to update the selected features and final-model hyperparameters.
- **Produced and checked the plots.** It added the `savefig` calls, noticed that the ROC curve was missing a `plt.show()`, viewed the saved images, and added feature-name labels to the heat map.

