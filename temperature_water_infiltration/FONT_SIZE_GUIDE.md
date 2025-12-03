# Font Size Customization Guide

## Multi-Pavement Comparison Plots

### Current Font Sizes (Updated)

All font sizes for the `plot_all_pavements_comparison()` function are now easily adjustable at the beginning of the function in `create_publication_figures.py`.

**Location in code:** Lines 196-201

```python
# ========== FONT SIZE SETTINGS (adjust here for all text elements) ==========
SUBPLOT_TITLE_SIZE = 20       # Size of pavement name titles (e.g., "Conventional Pavement")
AXIS_LABEL_SIZE = 18          # Size of axis labels (e.g., "Date", "Temperature (°C)")
TICK_LABEL_SIZE = 16          # Size of tick labels (numbers on axes)
LEGEND_SIZE = 16              # Size of legend text
RMSE_ANNOTATION_SIZE = 16     # Size of RMSE text annotation
# ===========================================================================
```

### What Each Setting Controls

1. **SUBPLOT_TITLE_SIZE (currently 20)**
   - Controls the pavement name at the top of each subplot
   - Examples: "CP Surface Temperature", "PICP Surface Temperature"
   - Applied to: `ax.set_title()`

2. **AXIS_LABEL_SIZE (currently 18)**
   - Controls the axis labels
   - "Date" on X-axis (bottom subplot only)
   - "Temperature (°C)" on Y-axis (all subplots)
   - Applied to: `ax.set_xlabel()` and `ax.set_ylabel()`

3. **TICK_LABEL_SIZE (currently 16)**
   - Controls the numbers on both axes
   - X-axis: Date values (e.g., "2023-10-05 12:00")
   - Y-axis: Temperature values (e.g., "10", "20", "30")
   - Applied to: `ax.tick_params()`

4. **LEGEND_SIZE (currently 16)**
   - Controls the legend text
   - Examples: "Observed Surface", "Modeled Surface", "Observed Water", "Modeled Water"
   - Applied to: `ax.legend()`

5. **RMSE_ANNOTATION_SIZE (currently 16)**
   - Controls the RMSE value annotation box
   - Example: "RMSE: 2.34°C"
   - Applied to: `ax.annotate()`

### How to Adjust Font Sizes

To change font sizes for all comparison plots:

1. Open `create_publication_figures.py`
2. Find the function `plot_all_pavements_comparison()` (around line 176)
3. Locate the "FONT SIZE SETTINGS" section (lines 196-201)
4. Modify the desired value(s)
5. Save the file
6. Run the script again: `python create_publication_figures.py`

**Example - Make fonts even larger:**
```python
# ========== FONT SIZE SETTINGS (adjust here for all text elements) ==========
SUBPLOT_TITLE_SIZE = 24       # Increased from 20
AXIS_LABEL_SIZE = 22          # Increased from 18
TICK_LABEL_SIZE = 20          # Increased from 16
LEGEND_SIZE = 18              # Increased from 16
RMSE_ANNOTATION_SIZE = 18     # Increased from 16
# ===========================================================================
```

**Example - Make fonts smaller:**
```python
# ========== FONT SIZE SETTINGS (adjust here for all text elements) ==========
SUBPLOT_TITLE_SIZE = 16       # Decreased from 20
AXIS_LABEL_SIZE = 14          # Decreased from 18
TICK_LABEL_SIZE = 12          # Decreased from 16
LEGEND_SIZE = 12              # Decreased from 16
RMSE_ANNOTATION_SIZE = 12     # Decreased from 16
# ===========================================================================
```

### Font Size Changes Made

**Previous values:**
- TITLE_SIZE = 14 (for subplot titles)
- LABEL_SIZE = 12 (for axis labels and legend)
- TICK_SIZE = 10 (for tick labels)

**New values (significantly increased):**
- SUBPLOT_TITLE_SIZE = 20 (+43% increase)
- AXIS_LABEL_SIZE = 18 (+50% increase)
- TICK_LABEL_SIZE = 16 (+60% increase)
- LEGEND_SIZE = 16 (+33% increase)
- RMSE_ANNOTATION_SIZE = 16 (+33% increase from previous hardcoded value of 12)

### Result

All comparison plots in `output/pavement_comparison_plots/` now have significantly larger and more readable text suitable for publication and presentations.

**Total plots updated:** 63 comparison plots (7-day intervals from May 2023 to July 2024)

---

*Last updated: 2025-11-30*
*All comparison plots regenerated with new font sizes*
