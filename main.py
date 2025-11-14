
import time

from dataset.dataset_loader import DatasetLoader
from ml_models.logistic_regression_model import LogisticRegressionModel
from ml_models.random_forest_model import RandomForestModel
from ml_models.svm_model import SVMModel


def main():
    loader = DatasetLoader()
    loader.load_all_datasets()
    
    datasets = ['iris', 'wine']
    for ds_name in datasets:
        info = loader.get_dataset_info(ds_name)
        print(f"  ✓ {info['name']}: {info['n_samples']} örnek, {info['n_features']} özellik, {info['n_classes']} sınıf")
    
    models = [
        LogisticRegressionModel(),
        RandomForestModel(n_estimators=100),
        SVMModel(kernel='rbf')
    ]
    
    for model in models:
        print(f"  ✓ {model.get_name()}")
    
    results = []
    
    for model in models:
        for dataset_name in datasets:
            print(f"\n  → {model.get_name()} + {dataset_name.upper()}")
            
            data = loader.get_dataset(dataset_name)
            dataset_info = loader.get_dataset_info(dataset_name)
            
            start_time = time.time()
            model.train(data['X_train'], data['y_train'])
            train_time = time.time() - start_time
            
            y_pred = model.predict(data['X_test'])
            
            cm = model.get_confusion_matrix(data['y_test'], y_pred)
            
            metrics = model.get_metrics(data['y_test'], y_pred)
            
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    Eğitim süresi: {train_time:.4f} saniye")
            
            results.append({
                'model_name': model.get_name(),
                'dataset_name': dataset_info['name'],
                'confusion_matrix': cm,
                'metrics': metrics,
                'class_names': dataset_info['classes'],
                'train_time': train_time
            })
    
    for dataset_name in datasets:
        dataset_results = [r for r in results if dataset_name in r['dataset_name'].lower()]
        best_result = max(dataset_results, key=lambda x: x['metrics']['accuracy'])
        
        print(f"\n{best_result['dataset_name']}:")
        print(f" Best: {best_result['model_name']}")
        print(f" Accuracy: {best_result['metrics']['accuracy']:.4f}")
        print(f" TimeZone: {best_result['train_time']:.4f} saniye")


if __name__ == "__main__":
    main()