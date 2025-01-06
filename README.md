<h1>Bonytics:  Pediatric Bone Age And Growth Analysis Using Deep Learning</h1>
<h2>Description</h2>
<p>This project focuses on leveraging deep learning techniques to predict bone age and classify bone density from hand X-ray images. 
The Bone Age Prediction module aids pediatricians and endocrinologists in estimating a child’s skeletal maturity, identifying growth plate status, and categorizing bone growth stages. 
Meanwhile, the Bone Density Classification module helps assess bone quality, identifying risks like osteoporosis or abnormal density levels.
</p>
<h2>Objectives</h2>
<ul>
<li>Bone Age Prediction and Distribution: Build a model to predict bone age from X-ray images and analyze the age distribution across groups.</li>
<li>Bone Growth Stages and Growth Plate Status: Classify bone growth stages and predict if growth plates are open, partially closed, or fully closed.</li>
<li>Bone Density Estimation: Estimate bone density using features from X-ray images</li>
</ul>
<h2>Implementation</h2>
<h2>1. Bone Age Prediction</h2>
<p>
<strong>Goal:</strong> Predict bone age from hand X-ray images and determine growth plate status and bone growth stage.
</p>
<p>
<strong>Steps:</strong>
<ul>
  <li>Data Preparation: RSNA Bone Age Dataset; preprocess images (resize to 256x256, normalize pixel values).</li>
  <li>Model: Xception (pre-trained on ImageNet) fine-tuned for regression.</li>
  <li>Output: Predict bone age as a continuous value, analyze growth plate status (open/closed), classify growth stages (early, middle, late adolescence).</li>
  <li>Evaluation: Use Mean Absolute Error (MAE) to compare predictions with ground truth.</li>
</ul>

<h2>2. Bone Density Classification</h2>
<p>
<strong>Goal:</strong> Classify hand X-rays into bone density categories (Low, Normal, High) to assess potential bone health issues.
</p>
<p>
<strong>Steps:</strong>
<ul>
  <li>Data Preparation: RSNA Bone Age Dataset; preprocess images (resize to 256x256, normalize pixel values).</li>
  <li>Feature Extraction: Use Xception (without top layers) for deep feature extraction.</li>
  <li>Clustering and Classification: Apply PCA for dimensionality reduction, KMeans for clustering into 3 density classes, and train a neural network classifier.</li>
  <li>Output Labels: Low Density (Osteoporosis), Normal Density, High Density.</li>
</ul>
