import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "6"   # or set to your CPU core count



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import load_breast_cancer


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # Matrix Approximation

 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning (Keras)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam


# Load Data 
# ---------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target) # 0 = Malignant, 1 = Benign

# Reverting to 1=Malignant, 0=Benign for standard medical convention
y = y.map({0: 1, 1: 0}) 

print(f"Dataset Shape: {X.shape}")
print("Class Distribution:\n", y.value_counts())

# Preprocessing & Scaling
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction (PCA)
# ---------------------------------------------------------
# Compressing 30 dimensions -> 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_train, palette='coolwarm', alpha=0.7)
plt.title('PCA Projection (Orthogonal Vector Space)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca_plot.png') 
print("PCA Plot saved as pca_plot.png")

# 4. Train Traditional Models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_acc = 0
best_model = None
best_pred = None
print("\n--- Model Evaluation ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_pred = preds
# Save the best model and scaler for the App
pickle.dump(best_model, open('best_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print(f"\nSaved {best_model} as 'best_model.pkl'")

cr = classification_report(y_test, preds)
print(cr)
cm = confusion_matrix(y_test, preds)
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model}')
plt.tight_layout()
plt.savefig(f'confusion_matrix.png')
plt.close()

print(f"Confusion Matrix for {best_model} saved as confusion_matrix_{best_model}.png")

print("\n--- Training Neural Network ---")

inputs = tf.keras.Input(shape=(30,))
x = Dense(32, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50, batch_size=32, verbose=0
)


loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Neural Network Accuracy: {acc:.4f}")


plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Neural Network Loss (Gradient Descent)')
plt.legend()
plt.savefig('nn_loss.png')
print("Neural Network Loss plot saved as nn_loss.png")
