import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore # Model oluşturma için
from tensorflow.keras.layers import Dense       # type: ignore # Dense katmanı için
from tensorflow.keras.optimizers import Adam    # type: ignore # Adam optimizer'ı için
from sklearn.preprocessing import StandardScaler

# Giriş verileri (Yaş ve Gelir problemleri olarak al)
X = np.array([[25, 2000], [30, 4000], [45, 10000], [50, 3000]])
y = np.array([[0], [1], [1], [0]])

# Basit model
model = Sequential()
model.add(Dense(6, input_shape=(2,), activation='relu'))  # input_dim yerine input_shape kullanılmalı
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
optimizer = Adam(learning_rate=0.005)  # optimizer doğru şekilde kullanılmalı
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Ortalama 0, varyans 1 olacak şekilde ölçeklendirme

# Modeli eğitme
model.fit(X_scaled, y, epochs=200, verbose=1)

# Tahmin etme
tahminleme = model.predict(X_scaled)
print("Tahminler:\n", tahminleme)

# Kullanıcıdan veri al ve kredi tahmini yap
while True:
    try:
        yaş = int(input("Yaşınızı giriniz: "))
        gelir = int(input("Gelirinizi giriniz: "))
    except ValueError:
        print("Lütfen geçerli bir sayı giriniz.")
        continue

    # Giriş verileri ölçeklendirme
    giriş_verisi = np.array([[yaş, gelir]])
    giriş_verisi_scaled = scaler.transform(giriş_verisi)

    # Tahmin oluştur
    prediction = model.predict(giriş_verisi_scaled)

    # sonucu yazdır
    if (prediction[0][0] < 0.50):
        print("Kredi verilmez")
    else:
        print("Kredi verilir")
    print(f"Tahmin: {prediction[0][0]:.4f}")

    devam = input("Başka bir tahmin yapmak ister misiniz? (evet/hayır): ").lower()
    if devam == 'hayır':
        print("İşlem  tamamlanmıştır.")

        break