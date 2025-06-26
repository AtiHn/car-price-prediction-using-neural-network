# Car Price Prediction Using Neural Networks 🚗🧠

This project predicts the **selling price of used cars** using a **deep neural network** built with TensorFlow and Keras.  
It includes data preprocessing, model training, evaluation, and visualization of learning metrics.

---

## 📁 Dataset

The dataset used is based on used car listings and contains features like:
- Fuel Type
- Transmission
- Owner
- Present Price
- Kms Driven
- Car Age  
and more...

We removed the `Car_Name` column and created a new feature: `car age`.

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- Pandas & NumPy
- Scikit-learn
- Matplotlib

---

## 🧠 Model Architecture

```text
Input Layer → Dense(64, ReLU)
             ↓
          Dense(32, ReLU)
             ↓
          Output (1 neuron)

Loss Function: Mean Squared Error (mse)
Optimizer: Adam
Evaluation Metric: Mean Absolute Error (mae)

---

## 🏁 How to Run This Project

1.Clone the repository:
git clone https://github.com/your-username/car-price-prediction-using-neural-network.git
cd car-price-prediction-using-neural-network

2. Install dependencies:
pip install -r requirements.txt

3. Run the main script:
python main.py

---

##📈 Sample Output

. Final Test MAE (example):
Mean Absolute Error on test data: 1.73

. Training & Validation Loss Graph

---

##🚧 Future Improvements

Hyperparameter tuning

Save and load the trained model (.h5)

Deploy the model as a web app using Streamlit or Flask

Feature importance analysis

---

##📦 Requirements

Here’s a sample requirements.txt you can include:
numpy
pandas
scikit-learn
tensorflow
matplotlib

You can save this as requirements.txt in the root folder.

---

