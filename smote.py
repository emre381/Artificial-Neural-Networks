import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential  # type: ignore # Model oluşturma için
from tensorflow.keras.layers import Dense       # type: ignore # Dense katmanı için
from tensorflow.keras.optimizers import Adam    # type: ignore # Adam optimizer'ı için
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

#  veri setini yükleme 
df = pd.read_excel('kredi_onay_verisi_1000.xlsx')
X= df[['Yaş','Gelir']].values
y=df[['Kredi Onayı']].values

# veriyi test ve eğitim olarak bölelim
X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2,random_state=42)
# smote ile veri dengesini sağlama
smote=SMOTE(random_state=42)
X_train_smote,y_train_smote = smote.fit_resample(X_train,y_train) #smote ile yapılan iş al verileri aldıktan sonra smote ile dengele dah dengeli bir şekle sok demek aslında 
# veriyi ölçeklendirme
scaler = StandardScaler()
X_train_smote_scaled= scaler.fit_transform(X_train_smote)
X_test_scaled=scaler.transform(X_test)
# modeli oluşturma
model=RandomForestClassifier(random_state=42)
model.fit(X_train_smote_scaled,y_train_smote)
# tahminleme 
y_pred=model.predict(X_test_scaled)
accuracy=accuracy_score(y_test,y_pred)

# yapay sinir ağları 
nn_model =Sequential()
nn_model.add(Dense(50,input_dim=2,activation='relu'))
nn_model.add(Dense(1,activation='sigmoid'))
# modeli  optimize et
optimizer =  Adam(learning_rate=0.001)
nn_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
# modeli eğitime al
nn_model.fit(X_train_smote_scaled,y_train_smote,batch_size=32,epochs= 75,verbose=1)

# yapay sinir ağı tahminkleme ve doğruluk oranı
y_pred_nn = (nn_model.predict(X_test_scaled)>0.5).astype('int32')
nn_accuracy=accuracy_score(y_test,y_pred_nn)
print(f"Test Doğruluk  Oranı: {accuracy*100:.2f}")

print(f"yapay sinir ağı doğruluk oranu:{nn_accuracy*100:.2f} %")


