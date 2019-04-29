# Earthquake Prediction

Authors: Elie Tetteh-Wayoe, Mihir Gadgil, Poornima Joshi

This is our attempt at the Los Alamos National Laboratory's [Kaggle competetion](https://www.kaggle.com/c/LANL-Earthquake-Prediction).

## Usage

1. The data is about 9 GB in size (2.3 GB compressed) and so has not been included in the repository.
- The file structure should be:

    .
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── test
    │   │   ├── seg_00030f.csv
    │   │   ├── ...
    │   └── train.csv
    ├── plot.png
    ├── quakes.ipynb
    └── quakes.py

2. Both the ipython notebook and the plain python file have mostly the same code.
3. Some code has been been commented out in both files to avoid reading huge amounts of data again and again.
4. The plotting code has been completely commented out of the plain python file, so it can be run without interruption.
5. The submission.csv file will be generated in the root directory and will OVERWRITE any pre-existing file.

