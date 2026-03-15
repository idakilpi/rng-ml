# RNG-ML

Machine learning–based classification of random number generator output using statistical features extracted from fixed-size byte windows.

## Research goal

This project investigates whether different random number generators can be distinguished from one another using machine learning based on statistical features extracted from their output.

## Research questions

1. Can weak pseudorandom number generators (PRNGs) be distinguished from cryptographically secure pseudorandom number generators (CSPRNGs) using machine learning based on statistical features?
2. Which statistical features are the most informative for distinguishing generator classes in the models used?
3. How do window size and feature set selection affect classification performance?

## Generator classes

- `weak`  
  Python-based linear congruential generator (`lcg`)
- `prng`  
  `mt`, `pcg64`
- `csprng`  
  `urandom`, `secrets`

## Models

The current implementation compares two supervised classification models:

- Logistic Regression
- Random Forest 

## Project structure

```text
rng-ml/
├── data/
│   ├── raw/
│   │   ├── lcg/
│   │   ├── mt/
│   │   ├── pcg64/
│   │   ├── secrets/
│   │   └── urandom/
│   ├── features/
│   │   └── features.parquet
│   └── results/
├── src/
│   ├── generate_streams.py
│   ├── extract_features.py
│   ├── train_models.py
│   └── plots.py
├── requirements.txt
└── README.md


