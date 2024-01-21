import tensorflow as tf
from keras.layers import Conv2D,Conv2DTranspose,BatchNormalization,Dense,Reshape,Flatten,MaxPooling2D
from keras.models import Model
import numpy as np 
import matplotlib.pyplot as plt
import keras.datasets.mnist

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape((60000,28,28,1)).astype('float32')/255.0

def build_generator():
    inp=keras.layers.Input(shape=(100,))
    model=Dense(7*7*128,activation='relu')(inp)
    model=Reshape((7,7,128))(model)
    model=BatchNormalization()(model)
    model=Conv2DTranspose(64,(3,3),strides=(2,2),padding='same',activation='relu')(model)
    model=BatchNormalization()(model)
    gen=Conv2DTranspose(1,(3,3),strides=(2,2),padding='same',activation='sigmoid')(model)
    
    return keras.models.Model(inp,gen)

def build_discriminator():
    image=keras.layers.Input(shape=(28,28,1))
    x=Conv2D(64,(3,3), activation='relu')(image)
    x=MaxPooling2D()(x)
    x=Conv2D(128,(3,3), activation='relu')(x)
    x=MaxPooling2D()(x)
    x=Flatten()(x)
    validity=Dense(1,activation='sigmoid')(x)
    return keras.models.Model(image, validity)

def build_gan(generator,discriminator):
    noise=keras.layers.Input(shape=(100,))
    gen_img=generator(noise)
    discriminator.trainable=False
    val=discriminator(gen_img)
    return keras.models.Model(noise,val)

discriminator=build_discriminator()
discriminator.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

generator=build_generator()
discriminator.trainable=False
gan=build_gan(generator,discriminator)
gan.compile(optimizer='adam',loss='binary_crossentropy')

epochs = 10000
batch_size = 512

for epoch in range(epochs):
    noise=np.random.normal(0, 1, size=[batch_size,100])
    gen_imgs=generator.predict(noise)
    idx=np.random.randint(0, x_train.shape[0],batch_size)
    real_imgs=x_train[idx]

    X=np.concatenate([real_imgs,gen_imgs])
    y_dis=np.zeros(2*batch_size)
    y_dis[:batch_size]=0.9

    d_loss=discriminator.train_on_batch(X, y_dis)

    noise=np.random.normal(0,1,size=[batch_size, 100])
    y_gen=np.ones(batch_size)
    g_loss=gan.train_on_batch(noise,y_gen)

    if epoch % 100==0:
        print(f"Epoch:{epoch},D loss:{d_loss}, G loss: {g_loss}")

generated_images=generator.predict(np.random.normal(0,1,size=[16,100]))
generated_images=generated_images.reshape(16, 28, 28)

plt.figure(figsize=(4,4))
for i in range(generated_images.shape[0]):
    plt.subplot(4,4,i+1)
    plt.imshow(generated_images[i],interpolation='nearest',cmap='gray_r')
    plt.axis('off')
plt.tight_layout()
plt.show()