# Heart Disease Prediction Model

## Quick Start
### Prerequisites
Python 3.8+ with packages listed in `requirements.txt`

### Installation
pip install -r requirements.txt

### Data Preparation
Place `cardio_base.csv` (semicolon-delimited) in the same directory as the script, or modify the file path in the source code.

### Execution
work.py

## Output
**Console output**: AUC score, classification report, top feature importance rankings

**Generated file**: `feature_importance.png` (feature importance visualization)

## Features
### Algorithm
LightGBM (Gradient Boosting Decision Tree)

### Feature Engineering
- Age conversion from days to years
- BMI calculation
- Pulse pressure calculation
- Blood pressure danger flag (systolic > 140 or diastolic > 90)

### Optimization Strategy
- Threshold moving (0.4 instead of default 0.5 to increase recall)
- ass weight adjustment (scale_pos_weight=1.2 to focus on positive cases)

## File Structure
- `heart_disease_predictor.py` - Main script
- `requirements.txt` - Python dependencies
- `cardio_base.csv` - Cardiovascular dataset (to be provided by user)
- `feature_importance.png` - Generated feature importance visualization

## Evaluation Metrics
Primary focus:
- **AUC**: Overall discriminative ability
- **Recall**: Maximize identification of positive cases
- **Feature interpretability**: Clear ranking of influential factors for medical insight

## Important Notes
- Dataset must comply with privacy standards (de-identified)
- Model is for assessment purposes only, not a replacement for professional medical diagnosis
- For public dissemination of results, follow Hack4Health rules regarding organizer notification

## Development
To modify or extend the model:
- Adjust classification threshold: modify `threshold = 0.4`
- Change model parameters: modify `clf = lgb.LGBMClassifier(...)` configuration
- Add features: include new column names in the `features` list
