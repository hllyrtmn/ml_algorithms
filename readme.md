# Machine Learning Comparison Project

This project tests 3 different machine learning algorithms on 2 different datasets and presents the results in an Excel report.

## Algorithms Used

1. **Logistic Regression** - Linear classification model
2. **Random Forest** - Ensemble learning method
3. **Support Vector Machine (SVM)** - Kernel-based classification

## Datasets

1. **Iris Dataset** - Classification of 3 different flower types (150 examples, 4 features)
2. **Wine Dataset** - Wine quality classification (178 examples, 13 features)

### Install Required Libraries

```bash
pip install -r requirements.txt
```
### Run the Program

```bash
python main.py
```
**Excel report**: The `ML_Diff_Report.xlsx` file is created

## Metrics

The following metrics are calculated for each model:

- **Accuracy**: Overall accuracy rate
- **Precision**: Accuracy of positive predictions
- **Recall**: Rate of capturing positive examples
- **F1-Score**: Harmonic mean of Precision and Recall

## Customization

You can change the parameters of the models in `main.py`:

```python
# More trees for Random Forest
RandomForestModel(n_estimators=200)

# Different kernel for SVM
SVMModel(kernel=‘linear’)  # or ‘poly’
```

### Test/Train Ratio

You can change the `test_size` value in `dataset_loader.py`:

```python
train_test_split(..., test_size=0.2, ...)  # 20% test
```

## Class Structure

### Model Classes
Each model class has the same structure:
- `train(X_train, y_train)`: Model training
- `predict(X_test)`: Making predictions
- `get_confusion_matrix(y_true, y_pred)`: CM calculation
- `get_metrics(y_true, y_pred)`: Calculate metrics
- `get_name()`: Return model name

### DatasetLoader Class
- `load_iris_dataset()`: Load Iris dataset
- `load_wine_dataset()`: Load Wine dataset
- `load_all_datasets()`: Load all datasets
- `get_dataset(name)`: Get a specific dataset
- `get_dataset_info(name)`: Get dataset information

### ExcelReporter Class
- `create_summary_sheet(results)`: Create a summary sheet
- `create_confusion_matrix_sheet(...)`: Create a CM sheet
- `add_algorithm_explanations_sheet()`: Add an explanation sheet
- `generate_report(results)`: Generate and save the report

## Notes

- Datasets are automatically normalized (StandardScaler)
- Train/test split ratio is 70%/30%
- Random state is fixed (42) for reproducible results
