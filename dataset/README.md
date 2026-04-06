Dataset folder structure (place images inside the class folders):

dataset/
  train/
    acne/
    rash/
    eczema/
    allergy/
    fungal/
  validation/
    acne/
    rash/
    eczema/
    allergy/
    fungal/
  test/
    acne/
    rash/
    eczema/
    allergy/
    fungal/

Recommended Kaggle datasets:
- 'Skin Lesion Analysis Towards Melanoma Detection' (useful for lesion pipeline)
- 'DermNet' or 'HAM10000' (for skin conditions dataset exploration)
- Search Kaggle for "skin disease classification", "dermatology images" and combine multiple small datasets.

Notes:
- Images should be RGB, preferably 224x224 or larger (we resize to 224x224 during preprocessing).
- Maintain a balanced number of images per class if possible.
- Keep file names unique and avoid nested subfolders other than class labels.
