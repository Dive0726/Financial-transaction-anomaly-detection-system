import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical
from flask import Flask, request, jsonify
import joblib
import warnings

warnings.filterwarnings('ignore')
from IPython.display import display


# 模拟数据生成函数（实际项目中替换为真实数据加载）
def generate_simulated_data(num_samples=10000, fraud_ratio=0.03):
    np.random.seed(42)

    # 基础交易特征
    data = {
        'amount': np.random.exponential(100, num_samples),
        'time': np.random.uniform(0, 24, num_samples),
        'age': np.random.randint(18, 70, num_samples),
        'gender': np.random.choice(['M', 'F'], num_samples),
        'category': np.random.choice(['grocery', 'electronics', 'clothing', 'travel', 'entertainment'], num_samples),
    }

    # 地理位置特征
    data['lat'] = np.random.normal(40, 5, num_samples)
    data['long'] = np.random.normal(-70, 5, num_samples)

    # 设备特征
    data['device_type'] = np.random.choice(['mobile', 'desktop', 'tablet'], num_samples)
    data['os'] = np.random.choice(['iOS', 'Android', 'Windows', 'MacOS'], num_samples)
    data['browser'] = np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'], num_samples)

    # 用户行为特征
    data['session_duration'] = np.random.exponential(300, num_samples)
    data['pages_visited'] = np.random.poisson(5, num_samples)
    data['previous_purchases'] = np.random.poisson(3, num_samples)

    # 生成标签 (0=正常, 1=欺诈)
    num_frauds = int(num_samples * fraud_ratio)
    data['is_fraud'] = np.concatenate([np.ones(num_frauds), np.zeros(num_samples - num_frauds)])
    np.random.shuffle(data['is_fraud'])

    # 为欺诈交易添加异常模式
    fraud_indices = np.where(data['is_fraud'] == 1)[0]
    data['amount'][fraud_indices] *= np.random.uniform(2, 10, len(fraud_indices))
    data['time'][fraud_indices] = np.random.uniform(0, 6, len(fraud_indices))  # 更多欺诈发生在凌晨
    data['lat'][fraud_indices] += np.random.normal(0, 10, len(fraud_indices))  # 地理位置突变
    data['long'][fraud_indices] += np.random.normal(0, 10, len(fraud_indices))
    data['session_duration'][fraud_indices] = np.random.exponential(30, len(fraud_indices))  # 欺诈会话通常较短

    return pd.DataFrame(data)


# 加载数据
df = generate_simulated_data(50000)
print(f"数据集形状: {df.shape}")
print(f"欺诈比例: {df['is_fraud'].mean():.2%}")

# 查看数据
display(df.head())
display(df.describe())


#数据清洗与预处理
def data_cleaning(df):
    # 复制原始数据
    df_clean = df.copy()

    # 处理缺失值 - 这里模拟数据没有缺失值，实际项目中可能需要
    # df_clean = df_clean.dropna()  # 或填充

    # 处理异常值 - 对数值特征进行Winsorization处理
    numeric_cols = ['amount', 'time', 'age', 'lat', 'long', 'session_duration', 'pages_visited', 'previous_purchases']

    for col in numeric_cols:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.05)
            q3 = df_clean[col].quantile(0.95)
            df_clean[col] = np.where(df_clean[col] < q1, q1, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > q3, q3, df_clean[col])

    # 类别型变量编码
    categorical_cols = ['gender', 'category', 'device_type', 'os', 'browser']
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    return df_clean


# 数据清洗
df_clean = data_cleaning(df)
print("清洗后数据形状:", df_clean.shape)



#特征工程
def feature_engineering(df):
    # 复制数据
    df_fe = df.copy()

    # 1. 交易金额相关特征
    df_fe['amount_log'] = np.log1p(df_fe['amount'])
    df_fe['amount_per_previous_purchase'] = df_fe['amount'] / (df_fe['previous_purchases'] + 1)

    # 2. 时间相关特征
    df_fe['hour_of_day'] = df_fe['time'] % 24
    df_fe['is_night'] = ((df_fe['hour_of_day'] >= 0) & (df_fe['hour_of_day'] <= 6)).astype(int)

    # 3. 地理位置特征
    # 这里简化处理，实际可以使用Haversine距离等计算位置变化
    df_fe['location_variation'] = np.sqrt(df_fe['lat'] ** 2 + df_fe['long'] ** 2)

    # 4. 用户行为特征
    df_fe['speed_per_page'] = df_fe['session_duration'] / (df_fe['pages_visited'] + 1)

    # 5. 交互特征
    df_fe['amount_age_ratio'] = df_fe['amount'] / df_fe['age']

    return df_fe


# 特征工程
df_fe = feature_engineering(df_clean)
print("特征工程后数据形状:", df_fe.shape)

# 分离特征和标签
X = df_fe.drop('is_fraud', axis=1)
y = df_fe['is_fraud']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 标准化数值特征
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
print(f"训练集欺诈比例: {y_train.mean():.2%}, 测试集欺诈比例: {y_test.mean():.2%}")


#无监督异常检测模型
#一：孤立森林 (Isolation Forest)
def isolation_forest_model(X_train, X_test, y_test):
    # 训练模型
    iso_forest = IsolationForest(n_estimators=150,
                                 max_samples='auto',
                                 contamination='auto',
                                 max_features=1.0,
                                 random_state=42)
    iso_forest.fit(X_train)

    # 预测异常得分
    test_scores = -iso_forest.score_samples(X_test)  # 转换为异常得分(越高越异常)

    # 评估模型 (将异常得分转换为标签)
    # 我们需要找到一个合适的阈值来将异常得分转换为欺诈/正常标签
    # 这里我们使用PR曲线找到最佳阈值
    precision, recall, thresholds = precision_recall_curve(y_test, test_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pared = (test_scores >= best_threshold).astype(int)

    # 评估指标
    print("Isolation Forest 性能:")
    print(classification_report(y_test, y_pared))

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    roc_auc = roc_auc_score(y_test, test_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'Isolation Forest (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Isolation')
    plt.legend()
    plt.show()

    return iso_forest


# 训练并评估孤立森林模型
iso_forest = isolation_forest_model(X_train, X_test, y_test)


#二：局部离群因子 (LOF)
def lof_model(X_train, X_test, y_test):
    # 训练模型
    lof = LocalOutlierFactor(n_neighbors=20,
                             contamination='auto',
                             novelty=True)  # novelty=True表示用于预测新样本

    # 由于LOF的novelty模式需要先fit训练数据
    lof.fit(X_train)

    # 预测异常得分
    test_scores = -lof.decision_function(X_test)  # 转换为异常得分(越高越异常)

    # 评估模型
    precision, recall, thresholds = precision_recall_curve(y_test, test_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (test_scores >= best_threshold).astype(int)

    print("LOF 性能:")
    print(classification_report(y_test, y_pred))

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    roc_auc = roc_auc_score(y_test, test_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'LOF (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LOF')
    plt.legend()
    plt.show()

    return lof


# 训练并评估LOF模型
lof = lof_model(X_train, X_test, y_test)


#有监督模型 (LightGBM)
#一：处理类别不平衡 (SMOTE)
def apply_smote(X_train, y_train):
    print("应用SMOTE前类别分布:", np.bincount(y_train))

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("应用SMOTE后类别分布:", np.bincount(y_train_smote))

    return X_train_smote, y_train_smote


# 应用SMOTE(必须调用SMOTE，以便提供给train_lightgbm函数调用数据）
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

#二：基础LightGBM模型
def train_lightgbm(X_train, y_train, X_test, y_test):
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 参数设置
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,  # 修改为-1减少输出
        'is_unbalance': True,  # 注意这里是is_unbalance不是is_unbalance
        'random_state': 42
    }

    # 使用callbacks替代evals_result
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)  # 每50轮输出一次日志
    ]

    # 训练模型
    gbm = lgb.train(params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[test_data],
                    callbacks=callbacks)

    # 预测
    y_pred_prob = gbm.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # 评估
    print("LightGBM 性能:")
    print(classification_report(y_test, y_pred))

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'LightGBM (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve before')
    plt.legend()
    plt.show()

    # 特征重要性
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(gbm, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()

    return gbm

# 应用SMOTE后调用LightGBM训练函数
gbm = train_lightgbm(X_train_smote, y_train_smote, X_test, y_test)
#模型优化 (贝叶斯优化)
def get_optimized_lightgbm():
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 45,
        'learning_rate': 0.08,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'random_state': 42,
        'verbose': -1  # 减少输出
    }

    train_data = lgb.Dataset(X_train_smote, label=y_train_smote)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 使用 callbacks 替代 early_stopping_rounds 和 verbose_eval
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)  # 每50轮打印一次日志
    ]

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        callbacks=callbacks  # 使用 callbacks 替代旧参数
    )

    return gbm

# 获取优化后的模型
optimized_lgb = get_optimized_lightgbm()

# 评估优化后的模型
y_pred_prob = optimized_lgb.predict(X_test, num_iteration=optimized_lgb.best_iteration)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("优化后的LightGBM 性能:")
print(classification_report(y_test, y_pred))

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f'Optimized LightGBM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve after')
plt.legend()
plt.show()

# 保存优化后的模型
joblib.dump(optimized_lgb, 'optimized_lightgbm_model.pkl')



# Flask API实现
app = Flask(__name__)

# 加载模型
model = joblib.load('optimized_lightgbm_model.pkl')


@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        # 获取请求数据
        data = request.get_json()

        # 转换为DataFrame
        input_data = pd.DataFrame([data])

        # 数据预处理 (与训练时相同的步骤)
        input_data_clean = data_cleaning(input_data)
        input_data_fe = feature_engineering(input_data_clean)

        # 确保列顺序与训练时一致
        # 在实际应用中，应该保存训练时的列顺序并在预测时保持一致
        # 这里简化处理
        missing_cols = set(X_train.columns) - set(input_data_fe.columns)
        for col in missing_cols:
            input_data_fe[col] = 0
        input_data_fe = input_data_fe[X_train.columns]

        # 标准化
        input_data_fe[numeric_cols] = scaler.transform(input_data_fe[numeric_cols])

        # 预测
        fraud_prob = model.predict(input_data_fe)[0]

        # 返回结果
        return jsonify({
            'status': 'success',
            'fraud_probability': float(fraud_prob),
            'is_fraud': int(fraud_prob >= 0.5)  # 使用0.5作为阈值
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


# 模拟API请求的函数
def simulate_api_request():
    # 创建一个测试交易
    test_transaction = {
        'amount': 350.0,
        'time': 3.5,  # 凌晨3:30
        'age': 28,
        'gender': 'M',
        'category': 'electronics',
        'lat': 45.0,
        'long': -75.0,
        'device_type': 'mobile',
        'os': 'iOS',
        'browser': 'Safari',
        'session_duration': 45,
        'pages_visited': 2,
        'previous_purchases': 1
    }

    with app.test_client() as client:
        response = client.post('/predict_fraud', json=test_transaction)
        return response.get_json()


# 启动API服务 (在实际部署中，这应该在单独的文件中)
if __name__ == '__main__':
    # 首先模拟一个API请求
    print("模拟API请求结果:")
    print(simulate_api_request())

    # 然后启动服务
    print("启动Flask服务...")
    app.run(host='0.0.0.0', port=5000, debug=True)