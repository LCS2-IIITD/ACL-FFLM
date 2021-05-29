from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# classifiers = {
#     'knn': KNeighborsClassifier(num_classes,n_jobs = -1),
#     'svm': SVC(kernel = "rbf", C = 1,random_state = 42,verbose = 1),
#     'random_forest': RandomForestClassifier(max_depth = 50, random_state = 42,n_jobs = -1),
#     'mlp': MLPClassifier(hidden_layer_sizes = (1024,),random_state = 42,early_stopping = True,validation_fraction = 0.1,shuffle = False,learning_rate_init = 0.001,learning_rate = 'adaptive',max_iter = 300,verbose = 1),
#     'decision_tree': DecisionTreeClassifier(max_depth = 5),
#     'gaussian_process':GaussianProcessClassifier(1.0 * RBF(1.0)),
#     'adaboost_classifier':AdaBoostClassifier(),
#     'naive_bayes':GaussianNB(),
#     'quadratic_discrim:':QuadraticDiscriminantAnalysis()
# }

def initialize_classifiers(num_classes, classifier_type = 'fast'):
    classifiers = {
        'svm': SVC(),
        'xgboost': XGBClassifier(objective = "multi:softprob"),
        'knn': KNeighborsClassifier(num_classes),
        'random_forest': RandomForestClassifier(),
        'mlp': MLPClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'naive_bayes':GaussianNB(),
        'quadratic_discrim:':QuadraticDiscriminantAnalysis(),
        'lda' : LinearDiscriminantAnalysis()
    }

    parameters = {
        'knn': {'n_neighbors':[5,10], 'weights': ['uniform','distance'], 'leaf_size' : [2,10],'n_jobs' : [-1]},
        'svm': {'kernel':['linear', 'rbf'], 'C':[0.1,1,10], 'gamma' : ['scale', 'auto'], 'random_state' : [42],'probability': [True]},
        'random_forest': {'n_estimators' : [10], 'max_depth' : [5,None], 'random_state' : [42],'criterion' : ['gini','entropy']},
        'mlp': {'hidden_layer_sizes' : [(64),(1024,)], 'random_state' : [42],'shuffle' : [False], 'learning_rate_init':[0.0001,0.01],'max_iter' : [50], 'learning_rate' : ['adaptive'], 'activation' : ['relu'], 'solver' :  ['adam'], 'alpha' : [0.0001,0.01], 'verbose' : [1], 'n_iter_no_change' : [10] },
        'decision_tree': {'criterion' : ['gini','entropy'], 'splitter' : ['best'], 'max_depth' : [5,None]},
        'naive_bayes': {},
        'quadratic_discrim:': {'reg_param':[0, 0.1, 0.5, 0.9]},
        'lda':{'solver' : ['lsqr'], 'shrinkage':['auto']},#'n_components':[16, 29]},
        #'lda':{'solver' : ['svd', 'lsqr', 'eigen'], 'shrinkage':['auto']},#'n_components':[16, 29]},
        'xgboost':{'booster':['gbtree', 'gblinear', 'dart'], 'max_depth' : [5,10], 'num_class':[num_classes],'nthread': [-1]}
        #'xgboost':{'booster':['gbtree'], 'max_depth' : [5], 'num_class':[num_classes],'nthread': [-1]}
    }
    
    random_parameters = {
        'knn': {'n_neighbors':5, 'weights': 'uniform', 'leaf_size' : 10, 'n_jobs' : -1},
        'svm': {'kernel':'rbf', 'C':0.1, 'gamma' : 'scale', 'random_state' : 42,'probability': True},
        'random_forest': {'n_estimators' : 10, 'max_depth' : 5, 'random_state' : 42,'criterion' : 'gini'},
        'mlp': {'hidden_layer_sizes' : 1024, 'random_state' : 42,'shuffle' : False, 'learning_rate_init':0.01,'max_iter' : 100, 'learning_rate' : 'adaptive', 'activation' : 'relu', 'solver' :  'adam', 'alpha' : 0.01},
        'decision_tree': {'criterion' : 'gini', 'splitter' : 'best', 'max_depth' : 5},
        'naive_bayes': {},
        'quadratic_discrim:': {'reg_param': 0.5},
        'lda':{'solver' : 'lsqr', 'shrinkage':'auto'},#'n_components':[16, 29]},
        #'lda':{'solver' : ['svd', 'lsqr', 'eigen'], 'shrinkage':['auto']},#'n_components':[16, 29]},
        'xgboost':{'booster':'gblinear', 'max_depth' : 5, 'num_class':num_classes,'nthread': -1}
        #'xgboost':{'booster':['gbtree'], 'max_depth' : [5], 'num_class':[num_classes],'nthread': [-1]}
    }
    
    
    
    if classifier_type == 'fast':
        parameters['svm']['kernel'] = ['linear']
        parameters.pop('knn')
        classifiers.pop('knn')
        parameters.pop('random_forest')
        classifiers.pop('random_forest')
        parameters.pop('xgboost')
        classifiers.pop('xgboost')
        parameters.pop('quadratic_discrim:')
        classifiers.pop('quadratic_discrim:')
        parameters.pop('svm')
        classifiers.pop('svm')
#         parameters.pop('mlp')
#         classifiers.pop('mlp')
        parameters.pop('decision_tree')
        classifiers.pop('decision_tree')
        parameters.pop('naive_bayes')
        classifiers.pop('naive_bayes')
    
    return classifiers, parameters

def new_classifier(classifier_name,num_classes,params):
    
    classifiers = {
    'xgboost': XGBClassifier(objective = "multi:softprob"),
    'lda': LinearDiscriminantAnalysis(**params),
    'knn': KNeighborsClassifier(**params),
    'svm': SVC(**params),
    'random_forest': RandomForestClassifier(**params),
    'mlp': MLPClassifier(**params),
    'decision_tree': DecisionTreeClassifier(**params),
    'adaboost_classifier':AdaBoostClassifier(**params),
    'naive_bayes':GaussianNB(**params),
    'quadratic_discrim:': QuadraticDiscriminantAnalysis(**params)
    }
    
    return classifiers[classifier_name]

if __name__ == '__main__':
    pass
    



    
    


