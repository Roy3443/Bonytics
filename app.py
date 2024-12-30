from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras import Model
import tensorflow as tf

app = Flask(__name__)

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Bone age statistics (mean and std from dataset)
mean_bone_age = 127.32
std_bone_age = 41.2

# Load the models and objects
@tf.keras.utils.register_keras_serializable()
def mae_in_months(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

bone_age_model = load_model('model/best_model.keras', custom_objects={'mae_in_months': mae_in_months})
bone_density_model = load_model('model/new_bone_density_model.keras')

# Load the scaler and PCA objects for bone density model
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the feature extraction model (e.g., Xception) for bone density features
def load_feature_extractor():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return feature_extractor

feature_extractor = load_feature_extractor()
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image for bone age model
        image = load_img(file_path, target_size=(256, 256))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Bone age prediction
        bone_age_prediction = bone_age_model.predict(image)
        bone_age_prediction = (bone_age_prediction[0][0] * std_bone_age) + mean_bone_age

        # Bone density prediction
        features = feature_extractor.predict(image)  # Extract features
        features_flat = features.reshape(1, -1)  # Flatten features
        features_scaled = scaler.transform(features_flat)  # Scale features
        features_pca = pca.transform(features_scaled)  # Apply PCA
        density_prediction = bone_density_model.predict(features_pca)
        predicted_class = np.argmax(density_prediction)
        class_mapping = {0: "Normal Density", 1: "Osteoporosis (Low Density)", 2: "High Density"}
        predicted_label = class_mapping[predicted_class]

        # Growth plate closure prediction logic
        if bone_age_prediction < 120:  # Example threshold for open plates
            growth_plate_status = "Open"
        elif bone_age_prediction < 180:  # Example threshold for partial closure
            growth_plate_status = "Partially Closed"
        else:  # Fully closed for higher bone ages
            growth_plate_status = "Closed"

        # Bone growth stage prediction logic
        if bone_age_prediction < 84:  # Example threshold for early stage (7 years)
            growth_stage = "Early Stage"
        elif bone_age_prediction < 168:  # Example threshold for mid stage (7-14 years)
            growth_stage = "Mid Stage"
        else:  # Late stage for higher bone ages
            growth_stage = "Late Stage"

        # Return the result page
        return render_template(
            'result.html',
            bone_age=round(bone_age_prediction, 2),
            bone_density=predicted_label,
            growth_plate=growth_plate_status,
            growth_stage=growth_stage,
            filename=filename
        )

    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)