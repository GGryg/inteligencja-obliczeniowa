# %%
opinion1 = "The hotel was excellent - super clean, quiet, good breakfast, and the staff were very friendly and helpful. The location was perfect, right in the center of town. I highly recommend this hotel!"
opinion2 = "This hotel was a complete disaster. The room was dirty and smelly, and the bed was uncomfortable. The staff were unhelpful and rude. I would never stay here again and I do not recommend it to anyone."

# %%
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

positive_scores = sid.polarity_scores(opinion1)
negative_scores = sid.polarity_scores(opinion2)

print("Score for positive: ", positive_scores)
print("Score for negative: ", negative_scores)

# %%
import text2emotion as t2e

positive_emotion = t2e.get_emotion(opinion1)
negative_emotion = t2e.get_emotion(opinion2)

print("Score for positive: ", positive_emotion)
print("Score for negative: ", negative_emotion)

# %% [markdown]
# Występuje problem z emoji w notatniku, rezulat jest z pliku zad2.py
# 
# Score for positive:  {'Happy': 0.5, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.5, 'Fear': 0.0}
# 
# Score for negative:  {'Happy': 0.0, 'Angry': 0.33, 'Surprise': 0.33, 'Sad': 0.17, 'Fear': 0.17}

# %% [markdown]
# Są częściowo zgodne z przewidzieniami
# 
# Porównanie
# 
# Vader prawidłowo rozpoznał choć nie chętnie, trochę ponad 50% dla pozytywnej, trochę ponad 30% dla negatywnej 
# 
# Text2Emotion dla pozytywnej opinii, choć rozpoznał ją po połowie, ale po drugiej rozpoznał ją jako smutną, jeśli chodzi o negatywną rozpoznał ją w 1/3 poprawnie, dodają przy tym zaskoczenie, smutek i starch 

# %%
opinion1 += " Spacious, bright, and veeeeery cosy! The view from the balcony was stunning and the bed was extremely comfortable! The greatest time in my life!"
opinion2 += " The bathroom was disgusting and the sheets were covered in stains. Could not sleep whole night, it was the worst night in my life."

# %%
positive_scores = sid.polarity_scores(opinion1)
negative_scores = sid.polarity_scores(opinion2)

print("Score for positive: ", positive_scores)
print("Score for negative: ", negative_scores)

# %%
positive_emotion = t2e.get_emotion(opinion1)
negative_emotion = t2e.get_emotion(opinion2)

print("Score for positive: ", positive_emotion)
print("Score for negative: ", negative_emotion)


