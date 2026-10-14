# Get the official C-MAPSS files

1. Open [NASA's C-MAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
2. Follow the `CMAPSSData.zip` resource link and download the archive.
3. Extract the text files directly into this repo's `data/raw/` directory (no nested archive folder).
4. For Phase 1, ensure these three files exist:
   - `data/raw/train_FD001.txt`
   - `data/raw/test_FD001.txt`
   - `data/raw/RUL_FD001.txt`
5. Run the data command documented in `README.md`.

Keep the original files unchanged. Files have no header and are separated by
whitespace. The loader assigns the 26 column names and checks the trajectories
and label counts. It stops with a clear error when a file is missing or malformed.
There is no automatic download and no synthetic replacement dataset. Tiny
artificial examples in `tests/` are exclusively unit-test fixtures.

The existing files supplied with this project are already in `data/raw/`.
