import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore # Model oluşturma için
from tensorflow.keras.layers import Dense       # type: ignore # Dense katmanı için

# giriş veriker (XOR problemleri olarak al)
X  = np.array([[0,0],[0,1],[1,0],[1,1]])
y   = np.array([[0],[1],[1],[0]])
# basit model
model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu')) # 4 kaynaktan yani 4 böron oluştur demek Dense input_dim 2 kaynağı check et yani X teki aksiyon yani relu doğruları bi yere yanlışları bir yere koy  
model.add(Dense(1,activation='sigmoid'))
# modeli derleme
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# modeli eğitme
model.fit(X, y, epochs=200, verbose=1) #epoch 100 0 ile 1 arasında 100 kere eğitim yap bunu modelini içine at model bunu öğrensin demek verbose her çıktıda bana cevap ver

# tahmin etme
tahminleme = model.predict(X)
print("Tahminler:\n",tahminleme)
