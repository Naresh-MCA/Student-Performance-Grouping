import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# Load the saved model
model = load_model('student_performance_model.h5')

# Load the saved encoders and scaler
with open('label_class.pkl', 'rb') as f:
    label_class = pickle.load(f)

with open('label_gender.pkl','rb') as f:
    label_gender=pickle.load(f)

with open('label_test.pkl','rb') as f:
    label_test=pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input data
input_data = {
    'gender': 'female',
    'parental level of education': 'some high school',
    'test preparation course': 'completed',
    'math score': 90,
    'reading score': 90,
    'writing score': 90,
    'average': 90
}
input_df = pd.DataFrame([input_data])
input_df

#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()
input_df['gender'] = label_gender.transform(input_df['gender']) 
input_df['test preparation course'] = label_test.transform(input_df['test preparation course']) 

one_hot = one_hot_encoder.transform([[input_data['parental level of education']]])
columns = one_hot_encoder.get_feature_names_out(['parental level of education'])
one_hot = pd.DataFrame(one_hot, columns=columns)

new_df1 = pd.concat([input_df.drop('parental level of education', axis=1), one_hot], axis=1)

from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
scaler_input=scaler.transform(new_df1)

prediction=model.predict(scaler_input)
predicted_class = np.argmax(prediction)

# Convert back to original label
final_label = label_class.inverse_transform([predicted_class])

print("Predicted Class:", final_label[0])
print("Prediction Probabilities:", prediction)