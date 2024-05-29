from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd 
from TransformationPipeline import TransformationPipeline
from LogScaling import LogScaling
from joblib import dump, load
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)


main_model = tf.keras.models.load_model('BestModels/CategoryModelV2.keras')

sub_models ={}
def sub_classify(main_class,img,image384):
    
    if(main_class == "Furniture"):
        sub_models["Furniture"] = tf.keras.models.load_model('BestModels/FurnitureModel.keras')
        predictions = sub_models["Furniture"].predict(img)
        class_names = ["Bed","Chair","Sofa","SwivelChair","Table"]
        i_max = np.argmax(predictions[0])
        predicted_class = class_names[i_max]
        return {"main_class" : "Furniture" , "sub_class": predicted_class}
    
    elif main_class == 'It':
        sub_models["It"] = tf.keras.models.load_model('BestModels/ItModelV1.keras')
        predictions = sub_models["It"].predict(img)
        class_names = ["Laptop","Printer","Samrtphone","Tv"]
        i_max = np.argmax(predictions[0])
        max = np.max(predictions[0])
        if(max<0.6):
            predicted_class = "Other"
        else:
            predicted_class = class_names[i_max]
        return {"main_class" : "It" , "sub_class": predicted_class}
    
    elif main_class == 'Jewellery':
        sub_models[main_class] = tf.keras.models.load_model('BestModels/JewelleryModelV1.keras')
        sub_models["Material"] = tf.keras.models.load_model('BestModels/MaterialModelV2.keras')
        jew_predictions = sub_models["Jewellery"].predict(img)
        mat_predictions = sub_models["Material"].predict(img)
        jew_class_names = ["earring","necklace","ring"]
        mat_class_names = ["gold","silver","bronze"]
        jew_max = np.argmax(jew_predictions[0])
        mat_max = np.argmax(mat_predictions[0])
        return {"main_class" : "Jewellery" , "sub_class": jew_class_names[jew_max] , "material": mat_class_names[mat_max] }
    
    elif main_class == 'Animal':
        sub_models[main_class] = tf.keras.models.load_model('BestModels/AnimalModel.keras')
        img_array = tf.keras.preprocessing.image.img_to_array(image384)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        predictions = sub_models[main_class].predict(img_array)
        decoded_prediction = tf.keras.applications.efficientnet_v2.decode_predictions(predictions, top=5)[0]
        return {"main_class" : "Animal" , "sub_class": decoded_prediction[0][1]}
    elif main_class == "Clothe":
        sub_models["Clothe"] = tf.keras.models.load_model('BestModels/ClothesModelV1.keras')
        predictions = sub_models["Clothe"].predict(img)
        class_names = ["Dress","Jacket","Pants","Shoes","Short","Suit","Sunglasses","T_shirt","Watch"]
        i_max = np.argmax(predictions[0])
        max = np.max(predictions[0])
        if(max<0.6):
            predicted_class = "Other"
        else:
            predicted_class = class_names[i_max]
        return {"main_class" : "Clothe" , "sub_class": predicted_class}
    else:
        return {"main_class": main_class}


#Image Classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Load and preprocess the image
        image = load_img(file_path, target_size=(224,224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255       

        image384 = load_img(file_path, target_size=(384,384))
        # Make predictions
        predictions = main_model.predict(image_array)
        print(predictions)
        class_names = ["Animal","Car","Clothe","Furniture","It","Jewellery"]
        i_max = np.argmax(predictions[0])
        max= np.max(predictions[0])
        if(max<0.69):
            predicted_class = "Other"
        else:
            predicted_class = class_names[i_max]
        
        result = sub_classify(predicted_class,image_array,image384)
        
        os.remove(file_path)
        
        return jsonify(result)


models = {}
models["catboost"] = load("LaptopPrice/Catboost.joblib")
models["lightgbm"] = load("LaptopPrice/Lightgbm.joblib")
preprocessor = load("LaptopPrice/preprocessor.joblib")

#Laptop Price Prediction
@app.route('/price/Laptop',methods=["Post"])
def predictPriceLaptop():
    columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit', 'graphic_card_gb', 'weight', 'warranty', 'Touchscreen', 'msoffice', 'rating', 'Number of Ratings', 'Number of Reviews']
    body = request.get_json()
    extracted_data = [body.get(column, None) for column in columns]
    X = pd.DataFrame([extracted_data],columns=columns)
    X = preprocessor.transform(X)
    pred1 = models['catboost'].predict(X)
    pred2 = models['lightgbm'].predict(X)
    final =  pred1*0.5+pred2*0.5
    return jsonify({"Price": final[0]})   


models["Mobile"] = load("MobilePrice/MobilePriceModel.joblib")
scaler = load("MobilePrice/scaler.joblib")
with open('MobilePrice/brands.json', 'r') as json_file:
    brands_str = json_file.read()
    brands_dict = json.loads(brands_str)
with open('MobilePrice/models.json', 'r') as json_file:
    models_str = json_file.read()
    models_dict = json.loads(models_str)    
#Mobile Price Prediction
@app.route('/price/Mobile',methods=["Post"])
def predictPriceMobile():
    body = request.get_json()
    if body["Model"] in models_dict:
        body["Model"] = models_dict[body["Model"]]
    else:
        body["Model"] = models_dict["OTHER"]
        
    if body["Brand"] in brands_dict:
        body["Brand"] = brands_dict[body["Brand"]]
    else:
        body["Brand"] = brands_dict["OTHER"]
    
    keys = ["Brand","Model","Storage","RAM","Screen Size","Battery","Camera"]
    values_list = [body[key] for key in keys]
    x = np.array([values_list],dtype=float)
    print(x)
    x = scaler.transform(x)      
    y = models["Mobile"].predict(x)  
    print(y[0])  
    return(jsonify({"Price":round(float(y[0]))}))

if __name__ == '__main__':
    app.run(debug=True)
