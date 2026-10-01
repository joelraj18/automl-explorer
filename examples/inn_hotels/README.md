# INN Hotels: predicting booking cancellations

`INNHotelsGroup.csv` has 36,275 hotel bookings. The column to predict is `booking_status` (32.8% Canceled).

| File | What it is |
|---|---|
| `INNHotels_cancellation_prediction.ipynb` | **The final project notebook**, already run so the outputs are saved |
| `build_notebook.py` | Generates the notebook and runs it: `python examples/inn_hotels/build_notebook.py` (needs `requirements-dev.txt`) |
| `INNHotelsGroup.csv` | The data |

## How it compares with the two earlier versions

| | Hand-made notebook | AutoML, no target chosen | **This notebook** |
|---|---|---|---|
| Task | classification ✅ | clustering ❌ (`booking_status` was used as an input) | classification ✅ |
| Duplicates | reported as 0, but `Booking_ID` hid them | – | **10,275 look-alike rows (28%)** found; models are also scored on unseen rows |
| Impossible dates | – | – | 37 bookings dated 29 Feb 2018, fixed; `arrival_weekday` added |
| Logistic regression | did not converge (exact-sum columns, and perfect separation on Complementary) | – | converges, max VIF 2.1, odds ratios explained in words |
| Tree pruning | `ccp_alpha` picked by **test** F1 (test-set leakage) | – | chosen by 5-fold cross-validation on the training data |
| Models | LR, 3 decision trees | KMeans | baseline, LR, 3 trees, random forest, gradient boosting |
| Best test F1 (Canceled) | 0.81 (optimistic) | – | **0.84** (random forest, chosen by CV); 0.74 on unseen rows |
| Insights | some numbers wrong (e.g. "Online ~40%", "Sep/Oct cancel most") | – | every number printed from the data |

The AutoML Explorer app now handles this file on its own:
- it pre-selects `booking_status` as the target
- it flags the look-alike rows and builds the weekday from the split date columns
- it compares logistic regression, a decision tree and a random forest
- it tunes the threshold

`tests/test_inn_hotels.py` checks all of this.
