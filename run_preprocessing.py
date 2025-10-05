from src.preprocess import run_preprocessing

# Alzheimer’s
run_preprocessing(
    "data/raw/alz",
    "data/processed/alz"
)

# Brain Tumor
run_preprocessing(
    "data/raw/bt",
    "data/processed/bt"
)
