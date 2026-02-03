# Scripts

Small helper utilities used during development and testing. These scripts are optional and do not affect the core pipeline in `src/`.

## Included

- `make_test_wireshark_csv.py`  
  Generates a synthetic Wireshark-style CSV (`No.`, `Time`, `Source`, `Length`) for reproducible end-to-end testing and demos. The output can be used as input to `src/signal.py`.
