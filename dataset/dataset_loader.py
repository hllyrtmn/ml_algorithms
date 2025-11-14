from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class DatasetLoader:
    
    def __init__(self):
        self.datasets = {
            'iris': None,
            'wine': None
        }
        self.dataset_info = {}
        
    def load_iris_dataset(self, test_size=0.3, random_state=42):
        """Iris dataset'ini yükle ve hazırla"""
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
        """Wine dataset'ini yükle ve hazırla"""
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
    
    def get_dataset(self, dataset_name):
        return self.datasets.get(dataset_name)
    
    def get_dataset_info(self, dataset_name):
        return self.dataset_info.get(dataset_name)
    
    def load_all_datasets(self):
        self.load_iris_dataset()
        self.load_wine_dataset()
        return self.datasets