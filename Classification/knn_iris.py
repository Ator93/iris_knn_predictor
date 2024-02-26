import requests
from collections import Counter
import csv
from typing import List, NamedTuple

Vector = List[float]

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

def parse_iris_row(row: List[str]) -> LabeledPoint:
    measurements = [float(value) for value in row[:-1]]
    label = row[-1].split("-")[-1]
    return LabeledPoint(measurements, label)

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key=lambda lp: sum((a - b) ** 2 for a, b in zip(lp.point, new_point)))
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)

# Download and parse the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = requests.get(url).text.strip().split('\n')
iris_data = [parse_iris_row(row.split(',')) for row in data]

# Prompt user for input and classify
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

new_point = [sepal_length, sepal_width, petal_length, petal_width]

k = 3  # Set the value of k for the k-NN classifier
predicted_class = knn_classify(k, iris_data, new_point)
print(f"The predicted class for the input is: {predicted_class}")