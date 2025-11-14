from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class DatasetLoader:
    
    def __init__(self):
        self.datasets = {
            'iris': None,
            'wine': None,
            'breast_cancer': None,
            'digits': None
        }
        self.dataset_info = {}
        
    def load_iris_dataset(self, test_size=0.3, random_state=42):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.datasets['iris'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': iris.feature_names,
            'target_names': iris.target_names
        }
        
        self.dataset_info['iris'] = {
            'name': 'Iris Dataset',
            'description': 'Çiçek türü sınıflandırması (3 sınıf)',
            'n_samples': len(iris.data),
            'n_features': len(iris.feature_names),
            'n_classes': len(iris.target_names),
            'classes': iris.target_names.tolist()
        }
        
        return self.datasets['iris']
    
    def load_wine_dataset(self, test_size=0.3, random_state=42):
        wine = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            wine.data, wine.target, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.datasets['wine'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': wine.feature_names,
            'target_names': wine.target_names
        }
        
        wine_class_names = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']
        
        self.dataset_info['wine'] = {
            'name': 'Wine Dataset',
            'description': 'Şarap kalitesi sınıflandırması (3 sınıf)',
            'n_samples': len(wine.data),
            'n_features': len(wine.feature_names),
            'n_classes': len(wine.target_names),
            'classes': wine_class_names
        }
        
        return self.datasets['wine']
    
    def load_breast_cancer_dataset(self, test_size=0.3, random_state=42):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        target_names = ['Malignant (Kötü Huylu)', 'Benign (İyi Huylu)']
        
        self.datasets['breast_cancer'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': cancer.feature_names,
            'target_names': target_names
        }
        
        self.dataset_info['breast_cancer'] = {
            'name': 'Breast Cancer Dataset',
            'description': 'Meme kanseri teşhisi (2 sınıf - kötü/iyi huylu)',
            'n_samples': len(cancer.data),
            'n_features': len(cancer.feature_names),
            'n_classes': len(target_names),
            'classes': target_names
        }
        
        return self.datasets['breast_cancer']
    
    def load_digits_dataset(self, test_size=0.3, random_state=42):
        
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        target_names = [f'Rakam {i}' for i in range(10)]
        
        self.datasets['digits'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [f'Pixel {i}' for i in range(64)],
            'target_names': target_names
        }
        
        self.dataset_info['digits'] = {
            'name': 'Digits Dataset',
            'description': 'El yazısı rakam tanıma (10 sınıf - 0-9 arası rakamlar)',
            'n_samples': len(digits.data),
            'n_features': 64,  # 8x8 pixel
            'n_classes': len(target_names),
            'classes': target_names
        }
        
        return self.datasets['digits']
    
    def get_dataset(self, dataset_name):
        return self.datasets.get(dataset_name)
    
    def get_dataset_info(self, dataset_name):
        return self.dataset_info.get(dataset_name)
    
    def load_all_datasets(self):
        self.load_iris_dataset()
        self.load_wine_dataset()
        self.load_breast_cancer_dataset()
        self.load_digits_dataset()
        return self.datasets
    
    def load_small_datasets(self):
        self.load_iris_dataset()
        self.load_wine_dataset()
        return ['iris', 'wine']
    
    def load_large_datasets(self):
        self.load_breast_cancer_dataset()
        self.load_digits_dataset()
        return ['breast_cancer', 'digits']
    
    def get_available_datasets_info(self):
        all_info = {
            'iris': {
                'name': 'Iris Dataset',
                'samples': 150,
                'features': 4,
                'classes': 3,
                'type': 'Küçük',
                'description': 'Çiçek türü sınıflandırması'
            },
            'wine': {
                'name': 'Wine Dataset',
                'samples': 178,
                'features': 13,
                'classes': 3,
                'type': 'Küçük',
                'description': 'Şarap kalitesi sınıflandırması'
            },
            'breast_cancer': {
                'name': 'Breast Cancer Dataset',
                'samples': 569,
                'features': 30,
                'classes': 2,
                'type': 'Orta',
                'description': 'Meme kanseri teşhisi'
            },
            'digits': {
                'name': 'Digits Dataset',
                'samples': 1797,
                'features': 64,
                'classes': 10,
                'type': 'Büyük',
                'description': 'El yazısı rakam tanıma (8x8 pixel)'
            }
        }
        return all_info