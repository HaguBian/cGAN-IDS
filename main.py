import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import CGAN
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df_train = pd.read_csv('dataset/KDDTrain.csv')
df_test = pd.read_csv('dataset/KDDTest.csv')

# Preprocessing data
df_train['attack'] = df_train['attack'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
df_test['attack'] = df_test['attack'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

features = ['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in', 'same_srv_rate',
            'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_serror_rate']

X_train = df_train[features]
y_train = df_train['attack']
X_test = df_test[features]
y_test = df_test['attack']

# Encode categorical features
for col in ['service', 'flag']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train ML models before CGAN
def train_ml_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'MLP': MLPClassifier(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
        print(f'{name} trained')
    return results

before_cgan_results = train_ml_models(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded)

# Train CGAN & Generate Synthetic Data
cgan = CGAN.CGAN(df_train, target_col='attack', latent_dim=100)
cgan.train(epochs=10000, batch_size=64)
df_synthetic = cgan.generate_synthetic_data(num_samples=2000)

# Train ML models after CGAN
X_augmented = np.vstack([X_train_scaled, scaler.transform(df_synthetic[features])])
y_augmented = np.hstack([y_train_encoded, label_encoder.transform(df_synthetic['attack'])])
after_cgan_results = train_ml_models(X_augmented, y_augmented, X_test_scaled, y_test_encoded)

print("Before CGAN:", before_cgan_results)
print("After CGAN:", after_cgan_results)

# Create and train the tuned CGAN
tuned_cgan = CGAN.TunedCGAN(X_train_scaled, y_train_encoded, latent_dim=128, learning_rate=0.0002, beta1=0.5)
tuned_cgan.train(epochs=10000, batch_size=128)

# Generate synthetic data
synthetic_X, synthetic_y = tuned_cgan.generate_synthetic_data(2000)
X_train_augmented = np.vstack((X_train_scaled, synthetic_X))
y_train_augmented = np.hstack((y_train_encoded, synthetic_y))

# Retrain ML models
after_tuned_cgan_results = train_ml_models(X_train_augmented, y_train_augmented, X_test_scaled, y_test_encoded)

# Compare results
print("Before CGAN:", before_cgan_results)
print("After Tuned_CGAN:", after_tuned_cgan_results)

# Data
ml_models = list(before_cgan_results.keys())
before_scores = np.array(list(before_cgan_results.values()))
after_scores = np.array(list(after_cgan_results.values()))
after_tuned_scores = np.array(list(after_tuned_cgan_results.values()))

score_diff_after = after_scores - before_scores
score_diff_tuned = after_tuned_scores - before_scores
score_diff_after_tuned = after_tuned_scores - after_scores

x = np.arange(len(ml_models))  # Positions for bars
width = 0.25  # Bar width

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Grouped Bar Chart - Before, After, and Tuned CGAN
ax1 = axes[0, 0]
bars1 = ax1.bar(x - width, before_scores, width, label="Before CGAN", color='red', alpha=0.7)
bars2 = ax1.bar(x, after_scores, width, label="After CGAN", color='blue', alpha=0.7)
bars3 = ax1.bar(x + width, after_tuned_scores, width, label="After Tuned CGAN", color='green', alpha=0.7)

ax1.set_ylabel("Accuracy")
ax1.set_title("Model Performance (Before, After, Tuned CGAN)")
ax1.set_xticks(x)
ax1.set_xticklabels(ml_models)
ax1.legend()
ax1.set_ylim(0.74, 0.82)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate values
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.3f}",
                 ha='center', va='bottom', fontsize=9, color='black')

# 2. Difference (After - Before)
ax2 = axes[0, 1]
bars = ax2.bar(ml_models, score_diff_after, color=['green' if diff > 0 else 'red' for diff in score_diff_after])
ax2.axhline(0, color='black', linewidth=1)
ax2.set_ylabel("Accuracy Change")
ax2.set_title("Performance Improvement (After - Before)")

for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}",
             ha='center', va='bottom', fontsize=10)

# 3. Difference (Tuned - After)
ax3 = axes[1, 0]
bars = ax3.bar(ml_models, score_diff_after_tuned, color=['green' if diff > 0 else 'red' for diff in score_diff_after_tuned])
ax3.axhline(0, color='black', linewidth=1)
ax3.set_ylabel("Accuracy Change")
ax3.set_title("Performance Improvement (Tuned - After)")

for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}",
             ha='center', va='bottom', fontsize=10)

# 4. Difference (Tuned - Before)
ax4 = axes[1, 1]
bars = ax4.bar(ml_models, score_diff_tuned, color=['green' if diff > 0 else 'red' for diff in score_diff_tuned])
ax4.axhline(0, color='black', linewidth=1)
ax4.set_ylabel("Accuracy Change")
ax4.set_title("Performance Improvement (Tuned - Before)")

for bar in bars:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}",
             ha='center', va='bottom', fontsize=10)

# Adjust layout and show plot
plt.tight_layout()
plt.show()