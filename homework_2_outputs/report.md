# Homework 2 Report

## Found Object Classes

- Class 1: period=1, count=1, feature_shape=(11, 11), positions=(22.5, 26.5)
- Class 2: period=1, count=1, feature_shape=(11, 11), positions=(30.5, 34.5)
- Class 3: period=1, count=1, feature_shape=(11, 11), positions=(16.5, 32.5)
- Class 4: period=1, count=2, feature_shape=(9, 9), positions=(14.7, 14.3), (18.6, 48.2)
- Class 5: period=2, count=1, feature_shape=(11, 11), positions=(16.5, 5.0)
- Class 6: period=1, count=2, feature_shape=(11, 11), positions=(30.5, 6.0), (36.9, 23.9)
- Class 7: period=1, count=1, feature_shape=(9, 9), positions=(2.0, 20.5)
- Class 8: period=1, count=1, feature_shape=(9, 9), positions=(47.8, 47.0)
- Class 9: period=1, count=2, feature_shape=(11, 11), positions=(40.5, 14.5), (10.5, 40.5)
- Class 10: period=1, count=1, feature_shape=(9, 9), positions=(8.8, 22.0)
- Class 11: period=1, count=1, feature_shape=(13, 13), positions=(6.5, 32.5)
- Class 12: period=1, count=1, feature_shape=(9, 9), positions=(43.2, 35.0)
- Class 13: period=1, count=1, feature_shape=(13, 13), positions=(35.7, 40.7)
- Class 14: period=1, count=1, feature_shape=(11, 11), positions=(23.9, 37.3)

## Model Comparison

- World size: 50x50
- Coverage: 1.0000
- Tracked objects: 18
- Grouped classes: 14
- Raster size: 2500 bits
- Object size without grouping: 2286 bits
- Object size with grouping: 1915 bits

## Accuracy vs Compression

- Hamming error without grouping: 0.057200
- Hamming error with grouping: 0.057200
- Best threshold from sweep: 20
- Error at best threshold: 0.046000
- Size at best threshold: 567 bits

## Conclusions

- The grouped object model is smaller than the raster map and also smaller than the ungrouped object model.
- Reconstruction is lossy because the model stores detected live-object structure, not the full background and not every segmentation detail.
- Lower grouping thresholds preserve more distinct classes, while larger thresholds compress more aggressively and can merge different objects.
- This solution does not model dynamic object motion prediction; it focuses on the required non-optional parts.

## Files

- `homework_2_outputs/error_over_time.png`
- `homework_2_outputs/threshold_tradeoff.png`
- `homework_2_outputs/model_size_comparison.png`
- `homework_2_outputs/model_comparison_table.md`
- `homework_2_outputs/timeline_metrics.csv`
- `homework_2_outputs/threshold_sweep.csv`
