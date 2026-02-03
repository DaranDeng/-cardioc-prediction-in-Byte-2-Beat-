import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb  # 引入新模型

# 1. 解决中文乱码问题 (为了画图)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False 

# --- 数据准备 (和你之前一样，保持不变) ---
df = pd.read_csv(r'E:\大创\黑客松\final\cardio_base.csv', sep=';')
df['age_years'] = (df['age'] / 365).round(1)
df = df[(df['ap_hi'] >= 40) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 180)]
df = df[df['ap_hi'] > df['ap_lo']]
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# 新增一个特征：血压是否极高 (危险信号)
df['bp_danger'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)

features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure', 'bp_danger']

X = df[features]
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. 模型升级：LightGBM ---
print("正在启动 LightGBM 训练...")
# scale_pos_weight=1.5 意思是稍微侧重“有病”的样本，防止漏诊
clf = lgb.LGBMClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    num_leaves=31, 
    scale_pos_weight=1.2, # 关键参数：增加对正样本的关注
    random_state=42,
    verbose=-1
)
clf.fit(X_train, y_train)

# --- 3. 策略优化：阈值移动 ---
# 预测概率而不是直接预测类别
y_prob = clf.predict_proba(X_test)[:, 1]

# 手动设定阈值：只要概率超过 0.4 就报警 (默认是0.5)
threshold = 0.4 
y_pred_adjusted = (y_prob >= threshold).astype(int)

# --- 4. 结果展示 ---
print("\n=== 优化后的成绩单 ===")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f} (稳定就是胜利)")
print("\n具体报告 (注意看 Recall 是否提升):")
print(classification_report(y_test, y_pred_adjusted))

# --- 5. 拓展功能：生成 PPT 素材图 ---
plt.figure(figsize=(10, 6))
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, X.columns)), columns=['Value','Feature'])

# 画条形图
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('黑客松模型：哪些因素最影响心脏健康？')
plt.tight_layout()
plt.savefig('feature_importance.png') # 保存图片
print("\n[完成] 图表已保存为 feature_importance.png，可直接放入 PPT。")

# 6. 验证结果
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]


# 7. 看看谁最重要 (给评委讲故事的核心)
feat_importances = pd.Series(clf.feature_importances_, index=features)
print("\n=== 影响心脏病最重要的 5 个因素 ===")

print(feat_importances.nlargest(5))
