"""ML Pipeline Pattern: chains the four components in order. Owner: Sultan.

Part of the KNN Workshop pipeline: DataLoader -> Preprocessor -> KNNModel -> Evaluator,
chained by KNNPipeline. Used by KNN_Workshop_Solution.ipynb.
"""

class KNNPipeline:
    def __init__(self, loader, preprocessor, model, evaluator):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model = model
        self.evaluator = evaluator

    def run(self):
        self.raw = self.loader.load()
        self.X, self.y = self.preprocessor.clean(self.raw)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split(self.X, self.y)
        X_train_scaled = self.preprocessor.fit_transform(self.X_train)
        X_test_scaled = self.preprocessor.transform(self.X_test)
        self.tuning = self.model.tune(X_train_scaled, self.y_train)
        self.model.fit(X_train_scaled, self.y_train)
        self.y_pred = self.model.predict(X_test_scaled)
        self.results = self.evaluator.compare({
            "Majority baseline": self.evaluator.baseline(self.y_train, self.y_test),
            f"KNN (k={self.model.best_k}, {self.model.best_metric})": self.evaluator.metrics(self.y_test, self.y_pred),
        })
        return self
