import pandas
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

ratings_dict ={
    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3]
}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

print(data.raw_ratings[:10])

sim_options = {
    "name" : "cosine",
    "user_based": False,
}

algo = KNNWithMeans(sim_options=sim_options)

trainingSet = data.build_full_trainset()
algo.fit(trainingSet)

prediction = algo.predict('E',2)
print(prediction.est)

