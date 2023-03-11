import os

import psycopg2
from psycopg2.extras import RealDictCursor

import pandas as pd

from catboost import CatBoostClassifier

conn = psycopg2.connect(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
    cursor_factory=RealDictCursor,
)

user_query = pd.read_sql('SELECT * FROM public.user_data', conn)
post_text_query = pd.read_sql('SELECT * FROM public.post_text_df', conn)
feed_data_query = pd.read_sql('SELECT * FROM public.feed_data limit 1000', conn)

user_df = pd.DataFrame(user_query)
post_df = pd.DataFrame(post_text_query)
feed_df = pd.DataFrame(feed_data_query)

united_data = feed_df.merge(post_df, on='post_id')

united_data = pd.merge(united_data, user_df, on='user_id')

ohe_cols = ['action', 'topic', 'country', 'city', 'os', 'source']

for col in ohe_cols:
    tmp = pd.get_dummies(united_data[col])
    united_data = pd.concat((united_data, tmp), axis=1)
    united_data.drop(col, axis=1, inplace=True)

united_data.drop(['user_id', 'timestamp', 'post_id', 'text'], axis=1, inplace=True)

X = united_data.drop('target', axis=1)
y = united_data['target']

cat = CatBoostClassifier()

cat.fit(X, y)

cat.save_model('catboost_model', format='cbm')


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/my/super/path")
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file
