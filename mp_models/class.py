import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

rseed = 42
num_char = 27 # 26 letters + space

num_epochs = 1500
learning_rate = 0.005
batch_s = 32

dat = pd.read_csv('datfull.csv')
dat.drop('Unnamed: 0', axis=1, inplace=True)
#dat['help']=dat['label'].values-97

x_dat_df=dat.drop('label', axis=1).values
print(x_dat_df.shape)
#x_dat_df=x_dat_df.reshape((x_dat_df.shape[0],int(x_dat_df.shape[1]/3),3))
#print(x_dat_df.shape)

#print(type(x_dat))
#print(x_dat)

y_dat_o=np.array(dat['label'])
y_dat_o=np.array([int(i) for i in y_dat_o])
y_dat_n=y_dat_o-97
y_dat_n=np.where((y_dat_n==32-97), 26, y_dat_n)
n_values = num_char
y_dat=np.eye(n_values)[y_dat_n]
print(y_dat[0:3])

print(type(y_dat))

#x_dat_df=dat.help.values


x_train, x_test, y_train, y_test = train_test_split(x_dat_df, y_dat, train_size=0.7, random_state=rseed)
#x_train = x_dat
#y_train = y_dat

x_train=tf.convert_to_tensor(x_train)
y_train=tf.convert_to_tensor(y_train)
x_test=tf.convert_to_tensor(x_test)
y_test=tf.convert_to_tensor(y_test)

print("CHKP1")


model = tf.keras.Sequential([
    tf.keras.layers.Input((21*3 ,)),

    #tf.keras.layers.Dense(128, activation='relu'),
    #,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_char, activation='softmax')
])


print("CHKP2")

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        #tf.keras.metrics.Precision(name='precision'),
        #tf.keras.metrics.Recall(name='recall')
    ]
)

print("CHKP3")

pred_pre = model.predict(x_test)
#print([i for i in zip(pred_pre,y_test)])
print([[np.argmax(i),np.argmax(j)] for i,j in zip(pred_pre,y_test)])

history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_s, validation_data=(x_test, y_test))

pred = model.predict(x_test)
print([[np.argmax(i),np.argmax(j)] for i,j in zip(pred,y_test)])


model.save('saved_models/newmodel')

print("CHKP4")
