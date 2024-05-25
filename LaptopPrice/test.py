import pandas as pd 
from TransformationPipeline import TransformationPipeline
from joblib import dump, load
from tensorflow.keras.models import load_model

preprocessor = load("preprocessor.joblib")
models = {}
models["ANN"] = load_model("ANN.keras")
models["catboost"] = load("Catboost.joblib")
models["lightgbm"] = load("Lightgbm.joblib")

a = ['ASUS', 'Intel', 'Core i5', '10th', '8 GB', 'DDR4', '512 GB', '0 GB', 'Windows', '32-bit', '2 GB', 'Casual', 'No warranty', 'No', 'No', '3 stars', 0, 0]

columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit', 'graphic_card_gb', 'weight', 'warranty', 'Touchscreen', 'msoffice', 'rating', 'Number of Ratings', 'Number of Reviews']

X = pd.DataFrame([a],columns=columns)

print(X)

X = preprocessor.transform(X)

pred1 = models["ANN"].predict(X.toarray())
pred2 = models['catboost'].predict(X)
pred3 = models['lightgbm'].predict(X) 

print(pred1*0.5+pred2*0.2+pred3*0.3)