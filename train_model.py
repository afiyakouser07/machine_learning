import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/user_behavior_dataset.csv")

le_gender = LabelEncoder()
le_os = LabelEncoder()
le_device = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Operating System"] = le_os.fit_transform(df["Operating System"])
df["Device Model"] = le_device.fit_transform(df["Device Model"])

X = df.drop(["User ID","User Behavior Class"], axis=1)
y = df["User Behavior Class"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

pickle.dump(model,
open("models/user_behavior_model.pkl","wb"))

print("Model Saved")
