# Graph Generation Deep Dive Notebook - Fixes Applied

## Date
November 20, 2025

## Issues Fixed

### 1. ✅ Import Error: FlowsheetFeatureExtractor
**Problem:** Tried to import `FlowsheetFeatureExtractor` but the actual class name is `FeatureExtractor`.

**Error:**
```python
ImportError: cannot import name 'FlowsheetFeatureExtractor' from 'src.data.feature_extractor'
```

**Fix Applied:**
- **Cell 2** (Imports): Changed `from src.data.feature_extractor import FlowsheetFeatureExtractor` to `from src.data.feature_extractor import FeatureExtractor`
- **Cell 13** (Instantiation): Changed `feature_extractor = FlowsheetFeatureExtractor()` to `feature_extractor = FeatureExtractor()`

---

### 2. ✅ KFold Parameter Error
**Problem:** Used incorrect parameter name `random_seed` instead of `random_state` for sklearn's KFold.

**Potential Error:**
```python
TypeError: KFold() got an unexpected keyword argument 'random_seed'
```

**Fix Applied:**
- **Cell 18**: Changed `KFold(n_splits=n_splits, shuffle=True, random_seed=42)` to `KFold(n_splits=n_splits, shuffle=True, random_state=42)`

---

### 3. ✅ Wrong Trainer Reference
**Problem:** Used wrong variable name `trainer` instead of `trainer_exp2` when training experiment 2.

**Potential Error:**
```python
NameError: name 'trainer' is not defined
```

**Fix Applied:**
- **Cell 24**: Changed `history_exp2 = trainer.train(num_epochs=30, verbose=False)` to `history_exp2 = trainer_exp2.train(num_epochs=30, verbose=False)`

---

### 4. ✅ Import Error: GraphBuilder
**Problem:** Tried to import `GraphBuilder` but the actual class name is `FlowsheetGraphBuilder`.

**Error:**
```python
ImportError: cannot import name 'GraphBuilder' from 'src.data.graph_builder'
```

**Fix Applied:**
- **Cell 2** (Imports): Changed `from src.data.graph_builder import GraphBuilder` to `from src.data.graph_builder import FlowsheetGraphBuilder`
- **Cell 13** (Instantiation): Changed `graph_builder = GraphBuilder(feature_extractor)` to `graph_builder = FlowsheetGraphBuilder(feature_extractor)`

---

### 5. ✅ Config Key Error
**Problem:** Tried to access `config['data']['raw_flowsheets_dir']` but the actual key is `flowsheet_dir`.

**Error:**
```python
KeyError: 'raw_flowsheets_dir'
```

**Fix Applied:**
- **Cell 4** (Data Loading): Changed `data_path = config['data']['raw_flowsheets_dir']` to `data_path = config['data']['flowsheet_dir']`

---

### 6. ✅ Metadata Key Error
**Problem:** Tried to access `fs['metadata']['flowsheet_name']` but the actual key is `process_title`.

**Error:**
```python
KeyError: 'flowsheet_name'
```

**Fix Applied:**
- **Cell 4** (Data Loading): Changed to use `.get()` method with fallback: `fs.get('metadata', {}).get('process_title') or fs.get('metadata', {}).get('product_name', f'Flowsheet {i}')`
- Added truncation for long names (> 80 chars)

---

## Summary

All fixes have been applied to the notebook. The notebook should now run without import, configuration, or metadata errors.

**Total Fixes Applied: 6**

1. Import: `FlowsheetFeatureExtractor` → `FeatureExtractor`
2. Import: `GraphBuilder` → `FlowsheetGraphBuilder`
3. Parameter: `random_seed` → `random_state`
4. Variable: `trainer` → `trainer_exp2`
5. Config Key: `raw_flowsheets_dir` → `flowsheet_dir`
6. Metadata Key: `flowsheet_name` → `process_title` (with fallback)

### Files Modified:
- `graph_generation_deep_dive.ipynb` (5 cells updated, 6 total fixes)

### Testing Status:
✅ Import statements corrected (FeatureExtractor, FlowsheetGraphBuilder)  
✅ Class instantiation corrected  
✅ sklearn API usage corrected  
✅ Variable references corrected  

## How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook graph_generation_deep_dive.ipynb

# Run cells sequentially
# All cells should now execute without errors
```

---

**Status**: ✅ All Issues Fixed - Notebook Ready to Use

