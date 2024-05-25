from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)


main_model = tf.keras.models.load_model('BestModels/CategoryModelV2.keras')

sub_models ={}
def sub_classify(main_class,img):
    
    if(main_class == "Furniture"):
        sub_models["Furniture"] = tf.keras.models.load_model('BestModels/FurnitureModel.keras')
        predictions = sub_models["Furniture"].predict(img)
        class_names = ["Bed","Chair","Sofa","SwivelChair","Table"]
        max = np.argmax(predictions[0])
        predicted_class = class_names[max]
        return {"main_class" : "Furniture" , "sub_class": predicted_class}
    
    elif main_class == 'It':
        sub_models["It"] = tf.keras.models.load_model('BestModels/ItModelV1.keras')
        predictions = sub_models["It"].predict(img)
        class_names = ["Laptop","Printer","Samrtphone","Tv"]
        max = np.argmax(predictions[0])
        if(max<0.8):
            predicted_class = "Other"
        else:
            predicted_class = class_names[max]
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
        predictions = sub_models[main_class].predict(img)
        decoded_prediction = tf.keras.applications.efficientnet_v2.decode_predictions(predictions, top=5)[0]
        return {"main_class" : "Animal" , "sub_class": decoded_prediction[0][1]}
    else:
        return {"main_class": main_class}


# Define a route to handle image upload and classification
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

        # Make predictions
        predictions = main_model.predict(image_array)
        class_names = ["Animal","Car","Clothe","Furniture","It","Jewellery"]
        max = np.argmax(predictions[0])
        if(max<0.7):
            predicted_class = "Other"
        else:
            predicted_class = class_names[max]
        
        result = sub_classify(predicted_class,image_array)
        
        os.remove(file_path)
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)