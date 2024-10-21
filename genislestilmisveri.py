import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential  # Model oluşturma için
from tensorflow.keras.layers import Dense       # Dense katmanı için
from tensorflow.keras.optimizers import Adam    # Adam optimizer'ı için
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veriyi oku
data = pd.read_excel('genisletilmisveri.xlsx')
df = pd.DataFrame(data)

# Kategorik verileri dummy değişkenlere dönüştür
df = pd.get_dummies(df, columns=["Medeni Durum", "Meslek", "Eğitim Durumu"], drop_first=True)

# Girdi ve çıktı verilerini ayır
X = df.drop("Kredi Onayı", axis=1).values
y = df["Kredi Onayı"].values

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur
model = Sequential()
model.add(Dense(25, input_dim=X.shape[1], activation='relu'))  # Girdi boyutunu dataframe'deki sütun sayısına göre ayarla
model.add(Dense(1, activation='sigmoid'))  # Çıktı 1 olacak, sigmoid aktivasyonu ile

# Modeli derle
optimizer = Adam(learning_rate=0.05)  # Öğrenme hızını belirle
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli eğit
model.fit(X_train_scaled, y_train, epochs=250, verbose=1)

# Tahmin yap
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)

# Doğruluk oranını hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Doğruluk Oranı : {accuracy * 100:.2f} %")

# Kullanıcıdan bilgi alıp tahmin yapma döngüsü
while True:
    yaş = int(input("Yaşınızı giriniz: "))
    gelir = int(input("Gelirinizi giriniz: "))
    medeni_durum = input("Medeni Durumunuzu giriniz (Bekar/Evli): ")
    meslek = input("Mesleğinizi giriniz (Mühendis/Doktor/Öğretmen/Avukat): ")
    eğitim_durumu = input("Eğitim Durumunuzu giriniz (Lisans/Yüksek Lisans/Doktora): ")

    # Kullanıcı verisini dummy değişkenlere uygun şekilde düzenle
    user_data = pd.DataFrame({'Yaş': [yaş], 'Gelir': [gelir], 
                              'Medeni Durum_Bekar': [1 if medeni_durum == "Bekar" else 0],
                              'Meslek_Mühendis': [1 if meslek == "Mühendis" else 0],
                              'Meslek_Doktor': [1 if meslek == "Doktor" else 0],
                              'Meslek_Öğretmen': [1 if meslek == "Öğretmen" else 0],
                              'Eğitim Durumu_Lisans': [1 if eğitim_durumu == "Lisans" else 0],
                              'Eğitim Durumu_Yüksek Lisans': [1 if eğitim_durumu == "Yüksek Lisans" else 0]})

    # Kullanıcı verilerini reindex ile orijinal veri çerçevesine uygun hale getir
    user_data = user_data.reindex(columns=df.drop('Kredi Onayı', axis=1).columns, fill_value=0)

    # Veriyi ölçeklendir
    user_data_scaled = scaler.transform(user_data)

    # Tahmin yap
    prediction = model.predict(user_data_scaled)
    print(f"Tahmin sonucu: {prediction[0][0]:.4f}")
