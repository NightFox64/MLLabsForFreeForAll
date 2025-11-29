import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator



def find_best_split(feature_values, target_labels):
    """
    Вычисляет оптимальный порог разделения по критерию Джини.
    Критерий определяется как:
    Q(R) = -(|R_l|/|R|) * H(R_l) - (|R_r|/|R|) * H(R_r)
    где:
    - R: исходная выборка
    - R_l, R_r: левая и правая подвыборки
    - H(S) = 1 - p₁² - p₀² (энтропия Джини)
    - p₁, p₀: доли объектов класса 1 и 0 в подвыборке
    """
    ### ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    sort_order = np.argsort(feature_values)
    sorted_features = feature_values[sort_order]
    sorted_labels = target_labels[sort_order]
    
    distinct_vals, first_occurrence = np.unique(sorted_features, return_index=True)
    
    if distinct_vals.size < 2:
        return np.array([]), np.array([]), None, None

    split_points = (distinct_vals[:-1] + distinct_vals[1:]) / 2.0
    
    total_samples = len(target_labels)
    total_positive = np.sum(target_labels)
    
    last_left_indices = first_occurrence[1:] - 1
    
    left_counts = last_left_indices + 1
    left_positives = np.cumsum(sorted_labels)[last_left_indices]
    
    left_positive_ratio = left_positives / left_counts
    left_negative_ratio = 1.0 - left_positive_ratio
    left_gini = 1.0 - (left_positive_ratio**2 + left_negative_ratio**2)
    
    right_counts = total_samples - left_counts
    right_positives = total_positive - left_positives
    
    right_positive_ratio = right_positives / right_counts
    right_negative_ratio = 1.0 - right_positive_ratio
    right_gini = 1.0 - (right_positive_ratio**2 + right_negative_ratio**2)
    
    gini_scores = - (left_counts / total_samples) * left_gini - (right_counts / total_samples) * right_gini
    
    best_idx = np.argmax(gini_scores)
    best_threshold = split_points[best_idx]
    best_score = gini_scores[best_idx]
    
    return split_points, gini_scores, best_threshold, best_score



class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._tree = {}

    def get_params(self, deep=True):
        """
        Метод, который возвращает словарь с параметрами, заданными в __init__.
        Нужен для sklearn.
        """
        return {
            "feature_types": self.feature_types,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf
        }

    def set_params(self, **params):
        """
        Метод для установки новых параметров. Нужен для sklearn.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth):
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self.max_depth is not None and depth >= self.max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self.min_samples_split is not None and len(sub_y) < self.min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best = None, None, None
        best_categories_map = {}

        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}

                sorted_categories = sorted(ratio.keys(), key=lambda k: ratio[k])
                categories_map = {category: i for i, category in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map.get(x) for x in sub_X[:, feature]])

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                if feature_type == "categorical":
                    best_categories_map = categories_map

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_type = self.feature_types[feature_best]
        if feature_type == "real":
            split = sub_X[:, feature_best] < threshold_best
        elif feature_type == "categorical":
            left_categories = {k for k, v in best_categories_map.items() if v < threshold_best}
            split = np.isin(sub_X[:, feature_best], list(left_categories))

        right_split = np.logical_not(split)

        if self.min_samples_leaf is not None and (
                np.sum(split) < self.min_samples_leaf or np.sum(right_split) < self.min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            left_categories = {k for k, v in best_categories_map.items() if v < threshold_best}
            node["categories_split"] = list(left_categories)

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_split], sub_y[right_split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self.feature_types[feature_idx]

        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, depth=0)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
