import pandas as pd
import numpy as np
import surprise
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

data = surprise.Dataset.load_builtin('ml-100k')

train, test = train_test_split(data, test_size=0.25, random_state=42)
algo = SVD()

algo.fit(train)

prediction = algo.test(test)
print(prediction)


df = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rate', 'id'])
del df['id']

df_table = df.set_index(["user", "item"]).unstack()


