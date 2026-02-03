# Data folder

This folder contains small CSV inputs used to run and demo the pipeline.

## Files

- `test.wireshark.csv`  
  A synthetic Wireshark-like CSV generated with the help of GPT for testing and demonstration. It is not real network traffic and is meant to validate the feature extraction and detection pipeline end-to-end.

- `Wireshark.csv`  
  A CSV export of my own normal network traffic captured on my machine (Wireshark → CSV). This file is used to test the system on real-world background traffic.

## Format

Both files follow the same minimal schema (Wireshark export mapped via `columns.json`):
- `No.` (packet index)
- `Time` (relative time in seconds)
- `Source` (source IP)
- `Length` (frame length in bytes)

## Note on sharing

Real captures may contain sensitive information (e.g., internal IPs). If this repository is shared publicly, consider excluding `Wireshark.csv` from version control and using only `test.wireshark.csv` for reproducible demos.

