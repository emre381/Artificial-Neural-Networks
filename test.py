import pandas as pd 
from tensorflow.keras.models import Sequential  # type: ignore # Model oluşturma için
from tensorflow.keras.layers import Dense       # type: ignore # Dense katmanı için
from tensorflow.keras.optimizers import Adam    # type: ignore # Adam optimizer'ı için
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data ={
'Yaş': [25, 30, 45, 50, 28, 35, 40, 60, 33, 48, 55, 29, 39, 53,
         42, 37, 44, 52, 58, 27, 49, 36, 41],
'Gelir': [2000, 4000, 10000, 3000, 3500, 5000, 8000, 12000, 4500,
           9500, 11000, 4200, 6200, 15000, 8700, 3100, 9800, 13000, 7000, 2900, 6400, 5200, 7200],
'Medeni Durum': ['Bekar', 'Evli', 'Evli', 'Bekar', 'Bekar', 'Evli',
                 'Evli', 'Bekar', 'Evli', 'Bekar', 'Evli', 'Bekar', 'Evli', 'Evli', 'Bekar', 'Evli', 'Bekar', 'Evli', 'Bekar', 'Evli', 'Evli', 'Bekar', 'Evli'],
'Meslek': ['Mühendis', 'Doktor', 'Öğretmen', 'Avukat', 'Mühendis',
           'Doktor', 'Avukat', 'Öğretmen', 'Doktor', 'Mühendis', 'Avukat', 'Mühendis', 'Öğretmen', 'Doktor', 'Avukat', 'Mühendis', 'Öğretmen', 'Doktor', 'Avukat', 'Mühendis', 'Öğretmen', 'Avukat', 'Mühendis'],
'Eğitim Durumu': ['Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora',
                  'Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora', 'Lisans', 'Yüksek Lisans', 'Doktora', 'Lisans', 'Yüksek Lisans', 'Doktora', 'Lisans',
                  'Lisans', 'Yüksek Lisans', 'Doktora', 'Yüksek Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora', 'Lisans'],
'Kredi Onayı': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
}
# pandas veri çerçevesi 
df = pd.DataFrame(data)
df.to_excel('genisletilmisveri.xlsx',index=False)