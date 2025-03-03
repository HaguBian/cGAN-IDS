# CGAN-Based Intrusion Detection System
This project implements a Conditional Generative Adversarial Network (CGAN) to enhance the performance of machine learning models in detecting network intrusions. The dataset used is the KDD dataset, and synthetic data generation is used to augment training data and improve classification accuracy.

This project is part of a university subject.
## Project Structure
```
├── dataset/
│   ├── KDDTrain.csv
│   ├── KDDTest.csv
├── main.py
├── CGAN.py
├── data_analysis.ipynb
├── README.md
```
### Files & Directories:
 - dataset/: Contains the KDDTrain.csv and KDDTest.csv datasets.
- main.py: The main script that loads data, trains models, and evaluates performance before and after CGAN augmentation.
- CGAN.py: Defines the CGAN model for generating synthetic data.
- data_analysis.ipynb: Jupyter notebook for exploratory data analysis and visualization.
- README.md: Project documentation.
## Requirements
To run this project, install the necessary dependencies using:
```
pip install -r requirements.txt
```
### Required Libraries:
- Python 3.11
- NumPy = 2.0.2
- Pandas = 2.2.3
- Matplotlib = 3.10.1
- Seaborn = 0.13.2
- Scikit-learn = 1.6.1
- TensorFlow = 2.18.0
## Usage
### 1. Run Data Analysis (Optional)
To explore and visualize the dataset, open data_analysis.ipynb in Jupyter Notebook:
```
jupyter notebook data_analysis.ipynb
```
### 2. Train and Evaluate Models
Run the main script to preprocess data, train models, and generate synthetic data:
```
python main.py
```
## Results
The models are trained and evaluated in three stages:
1. **Before CGAN**: Train machine learning models on the original dataset.
2. **After CGAN**: Augment the dataset with synthetic data and retrain models.
3. **After Tuned CGAN**: Use a tuned CGAN model for further performance improvement.
### Accuracy Results:
| Model          | Before CGAN | After CGAN | After Tuned CGAN |
|---------------|------------|------------|------------------|
| Random Forest | 0.7716     | 0.7730     | 0.7779          |
| SVM           | 0.7556     | 0.7552     | 0.7581          |
| MLP           | 0.7807     | 0.7749     | 0.8034          |
| Decision Tree | 0.8058     | 0.8081     | 0.8068          |

Evaluation metrics (accuracy) are plotted to compare performance at each stage.

## Observations

- **Random Forest** shows a slight improvement after applying CGAN and further tuning.
- **SVM** experiences almost no change with CGAN but sees a small improvement after tuning.
- **MLP** initially drops in performance after applying CGAN but significantly improves after tuning.
- **Decision Tree** improves after CGAN but sees a slight drop after tuning.
## Conclusion

The impact of CGAN varies across different models. While some models benefit from the additional synthetic data, others may require further tuning to optimize their performance.

## Contributing

Feel free to contribute by submitting issues or pull requests.
