from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class SVMModel:
    def __init__(self, kernel='rbf', random_state=42):
        self.model = SVC(kernel=kernel, random_state=random_state)
        self.model_name = "Support Vector Machine"
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
    
    def get_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def get_name(self):
        return self.model_name