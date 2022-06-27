import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

ml_df = pd.read_csv("ml_data.csv")
ml_df.drop(columns="Unnamed: 0", inplace=True)
ml_df.head()

x = np.array(ml_df["merged_text"])
y = np.array(ml_df["category"])

cnt_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))
x = cnt_vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = MultinomialNB()
model.fit(x_train,y_train)

def classify_article_category(text_to_predict):
    processed_data = cnt_vectorizer.transform([text_to_predict]).toarray()
    return model.predict(processed_data)
