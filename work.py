import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb  # Import LightGBM model

# 1. Fix Chinese font display issue for plotting
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set Chinese font
plt.rcParams['axes.unicode_minus'] = False  # Display minus signs correctly

# --- Data Preparation (unchanged) ---
df = pd.read_csv(r'E:\大创\黑客松\final\cardio_base.csv', sep=';')
# Convert age from days to years
df['age_years'] = (df['age'] / 365).round(1)
# Filter reasonable blood pressure ranges
df = df[(df['ap_hi'] >= 40) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 180)]
df = df[df['ap_hi'] > df['ap_lo']]  # Systolic should be higher than diastolic
# Calculate BMI
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
# Calculate pulse pressure
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Add new feature: blood pressure danger flag
df['bp_danger'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)

# Define feature set
features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure', 'bp_danger']

X = df[features]
y = df['cardio']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Model Upgrade: LightGBM ---
print("Starting LightGBM training...")
# scale_pos_weight=1.2 slightly weights the "diseased" samples to prevent missed diagnoses
clf = lgb.LGBMClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    num_leaves=31, 
    scale_pos_weight=1.2,  # Key parameter: increase focus on positive samples
    random_state=42,
    verbose=-1
)
clf.fit(X_train, y_train)

# --- 3. Strategy Optimization: Threshold Adjustment ---
# Predict probabilities instead of direct class labels
y_prob = clf.predict_proba(X_test)[:, 1]

# Manually set threshold: alarm if probability exceeds 0.4 (default is 0.5)
threshold = 0.4 
y_pred_adjusted = (y_prob >= threshold).astype(int)

# --- 4. Results Display ---
print("\n=== Optimized Performance Report ===")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nDetailed Classification Report (note recall improvement):")
print(classification_report(y_test, y_pred_adjusted))

# --- 5. Extended Function: Generate Visualization for Presentation ---
plt.figure(figsize=(10, 6))
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, X.columns)), columns=['Value','Feature'])

# Create bar plot
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Hackathon Model: Which Factors Most Impact Heart Health?')
plt.tight_layout()
plt.savefig('feature_importance.png')  # Save visualization
print("\n[Done] Chart saved as feature_importance.png, ready for presentation use.")

# 6. Additional validation
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# 7. Show top 5 most important features (key for storytelling to judges)
feat_importances = pd.Series(clf.feature_importances_, index=features)
print("\n=== Top 5 Most Influential Factors for Heart Disease ===")
print(feat_importances.nlargest(5))
