import pandas as pd
import numpy as np  
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# CGAN Class
class CGAN:
    """
    Conditional Generative Adversarial Network (CGAN) for generating synthetic data.

    This class implements a CGAN with the following steps:
    1. Data preprocessing:
       - The target column is label encoded.
       - Features are scaled using MinMaxScaler.
       - Categorical columns are label encoded.
    2. Model Building:
       - A generator and a discriminator model are built using Keras.
       - The generator takes random noise and class labels as input, generating synthetic samples.
       - The discriminator takes both real or fake samples and their corresponding labels as input, predicting whether they are real or fake.
       - A GAN model is built by stacking the generator and discriminator.

    Methods:
    - `train`: Trains the GAN for a specified number of epochs and batch size.
    - `generate_synthetic_data`: Generates synthetic data samples by passing random noise through the generator.

    Attributes:
    - `data`: The input data for training, including features and target column.
    - `target_col`: The target column name.
    - `latent_dim`: The size of the random noise input for the generator.
    - `features`: The list of feature column names.
    - `num_classes`: The number of unique target classes.
    - `X_train`: The scaled feature values.
    - `y_train`: The one-hot encoded target values.
    - `generator`: The generator model.
    - `discriminator`: The discriminator model.
    - `gan`: The combined GAN model.
    """
    def __init__(self, data, target_col, latent_dim=100):
        self.data = data.copy()
        self.target_col = target_col
        self.latent_dim = latent_dim
        self.features = [col for col in data.columns if col != target_col]
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data[target_col] = self.label_encoder.fit_transform(self.data[target_col])
        self.num_classes = len(self.label_encoder.classes_)
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        self.X_train = self.data[self.features].values
        self.y_train = to_categorical(self.data[target_col], num_classes=self.num_classes)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
    
    def build_generator(self):
        noise_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(self.num_classes,))
        x = Concatenate()([noise_input, label_input])
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(len(self.features), activation='tanh')(x)
        return Model([noise_input, label_input], x)
    
    def build_discriminator(self):
        feature_input = Input(shape=(len(self.features),))
        label_input = Input(shape=(self.num_classes,))
        x = Concatenate()([feature_input, label_input])
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model([feature_input, label_input], x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def build_gan(self):
        self.discriminator.trainable = False
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        fake_sample = self.generator([noise, label])
        validity = self.discriminator([fake_sample, label])
        model = Model([noise, label], validity)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
    
    def train(self, epochs=10000, batch_size=64):
        for epoch in range(epochs):
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            real_samples, real_labels = self.X_train[idx], self.y_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict([noise, real_labels])
            d_loss_real = self.discriminator.train_on_batch([real_samples, real_labels], np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch([fake_samples, real_labels], np.zeros((batch_size, 1)))
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_labels = np.ones((batch_size, 1))
            self.gan.train_on_batch([noise, real_labels], valid_labels)
    
    def generate_synthetic_data(self, num_samples=1000):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        fake_labels = np.eye(self.num_classes)[np.random.choice(self.num_classes, num_samples)]
        synthetic_data = self.generator.predict([noise, fake_labels])
        synthetic_data = self.scaler.inverse_transform(synthetic_data)
        df_synthetic = pd.DataFrame(synthetic_data, columns=self.features)
        df_synthetic[self.target_col] = self.label_encoder.inverse_transform(np.argmax(fake_labels, axis=1))
        return df_synthetic
    
# TunedCGAN Class
class TunedCGAN:
    """
    Tuned Conditional Generative Adversarial Network (CGAN) for generating synthetic data.

    Tuning Parameters:
    1. Learning Rate:
       - Value: 0.0002
    2. Beta1 Parameter for Adam Optimizer:
       - Value: 0.5
    3. Label Smoothing:
       - Value: 0.9 (for real labels)
    4. Latent Dimension (Noise Vector Size):
       - Value: 128
    5. Generator and Discriminator Architectures:
       - Generator Layers: Dense(256), Dense(512), followed by LeakyReLU activations and BatchNormalization.
       - Generator Output Activation: tanh
       - Discriminator Layers: Dense(512), Dense(256), followed by LeakyReLU activations.
       - Discriminator Output Activation: sigmoid
    6. Discriminator Loss Calculation:
       - Value: d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

    The class defines and trains a CGAN model with tuning to improve stability and performance.
    """
    def __init__(self, X_train, y_train, latent_dim=128, learning_rate=0.0002, beta1=0.5):
        self.X_train, self.y_train = X_train, tf.keras.utils.to_categorical(y_train)
        self.latent_dim = latent_dim
        self.num_classes = self.y_train.shape[1]
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Adam beta1 parameter for stability

        # Build models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        noise_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(self.num_classes,))
        x = Concatenate()([noise_input, label_input])
        
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.X_train.shape[1], activation='tanh')(x)
        return Model([noise_input, label_input], x)

    def build_discriminator(self):
        feature_input = Input(shape=(self.X_train.shape[1],))
        label_input = Input(shape=(self.num_classes,))
        x = Concatenate()([feature_input, label_input])
        
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, activation='sigmoid')(x)
        model = Model([feature_input, label_input], x)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, beta_1=self.beta1), 
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        fake_sample = self.generator([noise, label])
        validity = self.discriminator([fake_sample, label])
        model = Model([noise, label], validity)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, beta_1=self.beta1), loss='binary_crossentropy')
        return model

    def train(self, epochs=2000, batch_size=128):
        for epoch in range(epochs):
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            real_samples, real_labels = self.X_train[idx], self.y_train[idx]

            # Generate fake samples
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict([noise, real_labels])

            # Train Discriminator with smoothed labels
            real_y = np.ones((batch_size, 1)) * 0.9  # Smooth positive labels
            fake_y = np.zeros((batch_size, 1))  # Keep fake labels as 0

            d_loss_real = self.discriminator.train_on_batch([real_samples, real_labels], real_y)
            d_loss_fake = self.discriminator.train_on_batch([fake_samples, real_labels], fake_y)
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])  # Average loss

            # Train Generator
            g_loss = self.gan.train_on_batch([noise, real_labels], np.ones((batch_size, 1)))  # Wants to trick D

            if epoch % 200 == 0:
                print(f"Epoch {epoch}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

    def generate_synthetic_data(self, num_samples=1000):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        fake_labels = np.eye(self.num_classes)[np.random.choice(self.num_classes, num_samples)]
        synthetic_data = self.generator.predict([noise, fake_labels]) 
        return synthetic_data, np.argmax(fake_labels, axis=1)
