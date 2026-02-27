import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

#dosd data frame
df=pd.read_csv('StudentsPerformanceGrouping.csv')

df=df.drop(['lunch'],axis=1)

label_class=LabelEncoder()
label_gender=LabelEncoder()
label_test=LabelEncoder()
df['classgroup']=label_class.fit_transform(df['classgroup'])
df['gender']=label_gender.fit_transform(df['gender'])
df['test preparation course']=label_test.fit_transform(df['test preparation course'])

with open('label_class.pkl','wb') as f:
    pickle.dump(label_class,f)

with open('label_gender.pkl','wb') as f:
    pickle.dump(label_gender,f)

with open('label_test.pkl','wb') as f:
    pickle.dump(label_test,f)

# One-hot encoding
one_hot_encoder=OneHotEncoder(drop='first',sparse_output=False)
one_encoder=one_hot_encoder.fit_transform(df[['parental level of education']])

with open('one_hot_encoder.pkl','wb') as f:
    pickle.dump(one_hot_encoder,f)

columns=one_hot_encoder.get_feature_names_out(['parental level of education'])
#columns.head()

encoder_df=pd.DataFrame(one_encoder,columns=columns)
#encoder_df

new_df=pd.concat([df.drop('parental level of education',axis=1),encoder_df],axis=1)
#new_df.head()

x=new_df.drop(['classgroup'],axis=1)
y=new_df['classgroup']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)


#sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model=Sequential([
    Dense(64,activation='relu',input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dropout(0.2),
    Dense(16,activation='relu'),
    Dropout(0.2),
    Dense(3,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
log_dir='logs/fit'
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

early_stop=EarlyStopping(monitor='val_loss',
                         patience=10,
                         restore_best_weights=True)

history=model.fit(x_train,y_train,
                  validation_data=(x_test,y_test),
                  epochs=100,
                  batch_size=32,
                  callbacks=[tensorflow_callback,early_stop])


model.save('student_performance_model.keras')
#from tensorflow.keras.models import load_model
#%load_ext tensorboard


#%tensorboard --logdir logs/fit


