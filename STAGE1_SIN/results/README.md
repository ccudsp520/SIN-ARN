# Supplied historical results

These five CSV files were supplied with the original project and copied
without changing their contents. They were moved from the project root into
this directory. No experiment was rerun during packaging.

| File | Evaluations recorded |
| --- | --- |
| `table1_row1_results.csv` | One checkpoint, 12 conditions; legacy and XYZ metric columns |
| `table1_row1_epoch_sweep.csv` | E55, E56, E57, E58, E59, E60; 12 conditions each |
| `table1_row1_extended_epoch_sweep.csv` | E60, E65, E70, E73, E75, E80; 12 conditions each |
| `table1_row2_epoch_sweep.csv` | E60, E65, E70, E73, E75, E80; 12 conditions each |
| `table1_row3_epoch_sweep.csv` | E60, E65, E70, E73, E75, E80; 12 conditions each |

The supplied ZIP does not contain the evaluator that produced the Row 3 CSV.
The values in these files should retain their original metric labels and
experiment context. They are not a newly verified reproduction of the paper
or the complete E61-E80 checkpoint selection protocol.

Original scripts retain their original output paths. Set `OUTPUT_CSV` to a
new location before another evaluation to preserve these historical records.
