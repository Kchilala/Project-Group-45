# Urdu Digit Recognition

Project for predicting Urdu digits.

## Setup

This project uses `uv` for dependency management. If you don't have it, install it from [astral.sh/uv](https://astral.sh/uv).

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd project
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```



## Project Structure

- `data/`: Contains training and test images/CSVs.
- `urdu_digits/`: Custom library for data handling.
  - `dataset.py`: PyTorch `UrduDigitDataset` class.
  - `dataloader.py`: Helper for training/validation/test `DataLoader` creation.


## Roadmap for Further Development

### Data Augmentation & Preprocessing

### EDA

### Model Development

