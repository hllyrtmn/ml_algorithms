
import time

from dataset.dataset_loader import DatasetLoader
from ml_models.logistic_regression_model import LogisticRegressionModel
from ml_models.random_forest_model import RandomForestModel
from ml_models.svm_model import SVMModel
from report_generator.excel_report_service import ExcelReportService


def show_dataset_menu(loader):
    """Dataset seçim menüsünü göster"""
    print("\n" + "=" * 60)
    print("DATASET SEÇİMİ")
    print("=" * 60)
    
    available_datasets = loader.get_available_datasets_info()
    
    print("\nMevcut Dataset'ler:")
    print("-" * 60)
    for key, info in available_datasets.items():
        print(f"\n  📊 {info['name']}")
        print(f"     Örnekler: {info['samples']} | Özellikler: {info['features']} | Sınıflar: {info['classes']}")
        print(f"     Boyut: {info['type']} | {info['description']}")
    
    print("\n" + "-" * 60)
    print("\nSeçenekler:")
    print("  1 - Küçük Dataset'ler (Iris + Wine)")
    print("  2 - Büyük Dataset'ler (Breast Cancer + Digits)")
    print("  3 - Tüm Dataset'ler (Hepsi)")
    print("  4 - Manuel Seçim")
    
    while True:
        choice = input("\nSeçiminiz (1-4): ").strip()
        
        if choice == '1':
            print("\n✓ Küçük dataset'ler seçildi (Iris + Wine)")
            return loader.load_small_datasets()
        elif choice == '2':
            print("\n✓ Büyük dataset'ler seçildi (Breast Cancer + Digits)")
            return loader.load_large_datasets()
        elif choice == '3':
            print("\n✓ Tüm dataset'ler seçildi")
            loader.load_all_datasets()
            return ['iris', 'wine', 'breast_cancer', 'digits']
        elif choice == '4':
            selected = []
            print("\nDataset'leri seçin (virgülle ayırın):")
            print("  iris, wine, breast_cancer, digits")
            selection = input("Seçim: ").strip().lower()
            
            for ds in selection.split(','):
                ds = ds.strip()
                if ds in available_datasets:
                    selected.append(ds)
                    if ds == 'iris':
                        loader.load_iris_dataset()
                    elif ds == 'wine':
                        loader.load_wine_dataset()
                    elif ds == 'breast_cancer':
                        loader.load_breast_cancer_dataset()
                    elif ds == 'digits':
                        loader.load_digits_dataset()
            
            if selected:
                print(f"\n✓ Seçilen dataset'ler: {', '.join(selected)}")
                return selected
            else:
                print("Geçersiz seçim, tekrar deneyin.")
        else:
            print("Geçersiz seçim, lütfen 1-4 arası bir sayı girin.")


def main():

    loader = DatasetLoader()
    datasets = show_dataset_menu(loader)
    
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
    
    if len(datasets) == 2 and 'iris' in datasets and 'wine' in datasets:
        filename = 'ML_Diff_Report_Small_Datasets.xlsx'
    elif len(datasets) == 2 and 'breast_cancer' in datasets and 'digits' in datasets:
        filename = 'ML_Diff_Report_Big_Datasets.xlsx'
    elif len(datasets) == 4:
        filename = 'ML_Diff_Report_Full_Datasets.xlsx'
    else:
        filename = 'ML_Diff_Report_Special_Choice.xlsx'
    
    reporter = ExcelReportService(filename)
    report_file = reporter.generate_report(results)
if __name__ == "__main__":
    main()