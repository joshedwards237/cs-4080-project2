# Cache Performance Comparison: LRU vs LFU with Uniform Random Access Distrobution

This project implements and compares the performance of two popular caching algorithms: Least Recently Used (LRU) and Least Frequently Used (LFU) caches. The comparison is done using uniform random access patterns to evaluate their performance characteristics.

## Project Overview

The project implements both LRU and LFU cache algorithms and conducts extensive experiments to compare their performance across different parameters:

- Different cache sizes (as ratios of alphabet size)
- Various alphabet sizes
- Uniform random access patterns
- Performance metrics including hit rates, runtime, and memory usage

## Features

- Implementation of LRU and LFU cache algorithms
- Comprehensive performance testing framework
- Automated experiment execution with configurable parameters
- Detailed performance visualization and analysis
- Theoretical vs empirical performance comparison
- Output generation in multiple formats (CSV, PNG, TXT)

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - matplotlib
  - pandas
  - collections (built-in)

## Project Structure

```
.
├── josh_proj2.py          # Main implementation and experiment code
├── output/
│   ├── png/              # Generated performance plots
│   ├── csv/              # Raw experiment data
│   └── txt/              # Analysis summaries
└── README.md             # This file
```

## Key Findings

1. **Hit Rate Performance**:
   - Both LRU and LFU show similar hit rates for uniform random access patterns
   - Hit rates closely match theoretical predictions
   - Cache-to-alphabet ratio is the primary determinant of hit rate

2. **Runtime Performance**:
   - LRU demonstrates significantly better runtime performance
   - LFU runtime is approximately 233% higher than LRU

3. **Memory Usage**:
   - LFU requires about 62.5% more memory than LRU
   - This is due to additional frequency tracking overhead

## Usage

1. **Primary Method - Jupyter Notebook**:
   - Open the project in Jupyter Notebook
   - Run the cells in sequence to:
     - Execute experiments
     - Generate visualizations
     - View results interactively
   - This method allows for interactive exploration of results and easy modification of parameters

2. **Alternative Method - Python Script**:
   ```bash
   python josh_proj2.py
   ```
   The script will:
   - Create necessary output directories
   - Run experiments with different configurations
   - Generate performance visualizations
   - Save results in CSV format
   - Create a detailed analysis summary

## Output Files

- `output/png/`: Contains various performance comparison plots
- `output/csv/`: Contains raw experiment data and comparison results
- `output/txt/`: Contains detailed analysis summaries

## Recommendations

1. For uniform access distributions, LRU is generally preferable due to:
   - Lower runtime overhead
   - Lower memory usage
   - Similar hit rate performance

2. LFU may be more beneficial for:
   - Non-uniform distributions
   - Scenarios with frequently recurring items
   - When memory overhead is not a concern

## Theoretical vs Empirical Validation

The project includes validation against theoretical models:
- High correlation coefficients (>0.999) between theoretical and empirical results
- Detailed comparison of hit rate differences
- Runtime and memory usage analysis

## Contributing

This project was developed as part of CS 4080 - Advanced Algorithms. Contributions and improvements are welcome.

## License

This project is part of an academic course and should be used in accordance with academic integrity policies.