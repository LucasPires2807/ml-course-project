from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn import metrics

class ModelWrapper:
    
    def __init__(self, algorithm_class, features, classes, **algorithm_params):
        """
        Initializes the ModelWrapper with the chosen algorithm, features, and classes.

        Parameters:
        - algorithm_class: the class of the algorithm (e.g., KNeighborsClassifier, DecisionTreeClassifier)
        - features: array-like, feature dataset
        - classes: array-like, target dataset
        - **algorithm_params: additional parameters for initializing the algorithm
        """
        self.__algorithm = algorithm_class(**algorithm_params)
        self.__features = features
        self.__classes = classes
        self.metrics = None

    def __holdout(self, **kwargs):
        # Get test_size and random_state from kwargs, with defaults if not provided
        test_size = kwargs.get('test_size', 0.3)
        random_state = kwargs.get('random_state', 1)
        
        # Perform holdout split
        X_train, X_test, y_train, y_test = train_test_split(
            self.__features, self.__classes, test_size=test_size, random_state=random_state
        )
        self.__algorithm.fit(X_train, y_train)
        y_pred = self.__algorithm.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Holdout (70-30) Split Accuracy:", accuracy)
        self.metrics = accuracy

    def __kfold(self, **kwargs):
        # Get n_splits and random_state from kwargs, with defaults if not provided
        n_splits = kwargs.get('n_splits', 10)
        random_state = kwargs.get('random_state', 1)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(self.__algorithm, self.__features, self.__classes, cv=kf)
        print("K-Fold Cross-Validation Accuracy:", scores.mean())
        self.metrics = scores.mean()

    def model_train(self, method, **kwargs):
        """
        Trains the model based on the specified method.

        Parameters:
        - method: str, either 'holdout' or 'kfold'
        - **kwargs: additional parameters for the holdout or kfold methods (e.g., test_size, n_splits)
        
        Returns:
        - metrics: Accuracy score for holdout or mean and std for kfold
        """
        match method:
            case 'holdout':
                self.__holdout(**kwargs)
            case 'kfold':
                self.__kfold(**kwargs)
            case _:
                print("Method should be 'holdout' or 'kfold'.")

        return self.metrics
