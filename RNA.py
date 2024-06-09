import tensorflow as tf
import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(imagens_treinamento, imagens_treinamento_labels), (imagens_teste, imagens_teste_labels) = fashion_mnist.load_data()


'''plt.imshow(imagens_treinamento[1], cmap=plt.cm.binary,)
plt.show()'''

#print(imagens_treinamento[1])

nomes_das_classes = ['Camiseta/Top', 'Calça', 'Suéter', 'Vestido', 'Casaco','Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Botas']

imagens_treinamento = imagens_treinamento / 255.0
imagens_teste = imagens_teste / 255.0

'''plt.figure(figsize=(20,20))
plt.rcParams.update({'font.size': 18})
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.imshow(imagens_treinamento[i], cmap=plt.cm.binary)
    plt.xlabel(nomes_das_classes[imagens_treinamento_labels[i]])
plt.show()'''

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

modelo.fit(imagens_treinamento, imagens_treinamento_labels, epochs=10)

valor_perda, valor_precisao = modelo.evaluate(imagens_teste, imagens_teste_labels)
print('\nPrecisão na classificação das amostras de teste:', valor_precisao)

valor_perda_treinamento, valor_precisao_treinamento = modelo.evaluate(imagens_treinamento, imagens_treinamento_labels)
print('\nPrecisão na classificação das amostras de treinamento:', valor_precisao_treinamento)

classificacao = modelo.predict(imagens_teste)

plt.figure(figsize=(30,30))
for i in range(12):
    plt.subplot(5,3,i+1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagens_teste[i], cmap=plt.cm.binary)
    plt.xlabel("\nFigura utilizada: "+ nomes_das_classes[imagens_teste_labels[i]])
    plt.title("Classe reconhecida: "+ nomes_das_classes[np.argmax(classificacao[i])])
plt.show()