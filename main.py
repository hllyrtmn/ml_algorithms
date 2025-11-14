
import time
from dataset.dataset_loader import DatasetLoader
from ml_models.logistic_regression_model import LogisticRegressionModel
from ml_models.random_forest_model import RandomForestModel
from ml_models.svm_model import SVMModel
from report_generator.excel_report_service import ExcelReportService


def main():
    loader = DatasetLoader()
    loader.load_all_datasets()
    
    datasets = ['iris', 'wine']

    models = [
        LogisticRegressionModel(),
        RandomForestModel(n_estimators=100),
        SVMModel(kernel='rbf')
    ]
    
    results = []
    
    for model in models:
        for dataset_name in datasets:
            
            data = loader.get_dataset(dataset_name)
            dataset_info = loader.get_dataset_info(dataset_name)
            
            start_time = time.time()
            model.train(data['X_train'], data['y_train'])
            train_time = time.time() - start_time
            
            y_pred = model.predict(data['X_test'])
            
            cm = model.get_confusion_matrix(data['y_test'], y_pred)
            
            metrics = model.get_metrics(data['y_test'], y_pred)
            
            results.append({
                'model_name': model.get_name(),
                'dataset_name': dataset_info['name'],
                'confusion_matrix': cm,
                'metrics': metrics,
                'class_names': dataset_info['classes'],
                'train_time': train_time
            })
    
    report_service = ExcelReportService('ML_Diff_Report.xlsx')
    report_file = report_service.generate_report(results)


if __name__ == "__main__":
    main()