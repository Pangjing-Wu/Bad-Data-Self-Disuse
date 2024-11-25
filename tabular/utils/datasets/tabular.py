import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Diabetes130(object):

    def __init__(self, root: str) -> None:
        self.data = pd.read_csv(root)
        self.label_mapping = {'NO': 0, '>30': 1, '<30': 1}
        self.X = self.data.drop(['readmitted', 'encounter_id', 'patient_nbr'], axis=1)
        self.y = self.data['readmitted'].map(self.label_mapping)
        for column in self.X.select_dtypes(include=['object']).columns:
            self.X[column] = LabelEncoder().fit_transform(self.X[column])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    @property
    def trainset(self):
        return self.X_train, self.y_train
    
    @property
    def testset(self):
        return self.X_test, self.y_test
    
    
class Adults(object):

    def __init__(self, root: str) -> None:
        self.data = pd.read_csv(root)
        self.X = self.data.drop('income', axis=1)
        self.y = self.data['income']
        for column in self.X.select_dtypes(include=['object']).columns:
            self.X[column] = LabelEncoder().fit_transform(self.X[column])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
    @property
    def trainset(self):
        return self.X_train, self.y_train
    
    @property
    def testset(self):
        return self.X_test, self.y_test