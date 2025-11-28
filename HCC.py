import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# 1. 加载模型 & 数据
# -----------------------------
# TODO：这里改成你实际的模型文件名，如 "HCC_SVM.pkl"
MODEL_PATH = "XGBoost.pkl"
DATA_PATH = "HCC.csv"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

model = load_model()
data = load_data()

# 特征名（按 HCC.csv 中除 group 外的所有列）
feature_names = [
    'original_shape_Elongation_A',
    'log.sigma.1.0.mm.3D_glcm_Correlation_A',
    'lbp.3D.k_gldm_DependenceEntropy_A',
    'original_glcm_Autocorrelation',
    'original_gldm_LargeDependenceHighGrayLevelEmphasis',
    'log.sigma.1.0.mm.3D_firstorder_Median',
    'log.sigma.1.0.mm.3D_glrlm_ShortRunHighGrayLevelEmphasis',
    'wavelet.HHL_ngtdm_Complexity',
    'exponential_glcm_InverseVariance',
    'Alcohol',
    'AFP',
    'ALT',
    'AST'
]

X_all = data[feature_names]

# 预计算每个特征的 min / max / median，方便给 number_input 用
stats = X_all.agg(['min', 'max', 'median'])

# -----------------------------
# 2. 构建 SHAP 解释器（KernelExplainer，兼容 SVM/Logistic 等）
# -----------------------------
def build_shap_explainer(model, X_background):
    """
    使用部分样本作为背景数据，构建 KernelExplainer。
    f(x) 返回阳性（group=1）的预测概率。
    """
    # 随机采样一部分样本作为背景（避免太慢）
    background = X_background.sample(
        n=min(50, len(X_background)),
        random_state=0
    )

    # 定义返回阳性概率的预测函数
    def f(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(f, background)
    return explainer

explainer = build_shap_explainer(model, X_all)

# -----------------------------
# 3. Streamlit 页面布局
# -----------------------------
st.title("HCC 疗效预测在线计算器（联合模型）")

st.markdown("""
本工具基于机器学习模型，对接受特定治疗方案的肝细胞癌（HCC）患者进行**疗效（如客观缓解 vs 非缓解）预测**。  
输入患者的临床指标及影像组学特征后，模型将给出达到 **“良好疗效（如 CR/PR）”** 的概率，并提供 SHAP 可解释图。
> ⚠️ 本计算器仅用于科研和教学目的，不能替代临床医生的专业判断。
""")

st.subheader("1. 输入变量")

st.markdown("**（1）影像组学特征（已标准化的 Z-score 值）**")

# radiomics 特征输入：使用数据中的 min/max/median，step 0.01
input_values = {}

for col in [
    'original_shape_Elongation_A',
    'log.sigma.1.0.mm.3D_glcm_Correlation_A',
    'lbp.3D.k_gldm_DependenceEntropy_A',
    'original_glcm_Autocorrelation',
    'original_gldm_LargeDependenceHighGrayLevelEmphasis',
    'log.sigma.1.0.mm.3D_firstorder_Median',
    'log.sigma.1.0.mm.3D_glrlm_ShortRunHighGrayLevelEmphasis',
    'wavelet.HHL_ngtdm_Complexity',
    'exponential_glcm_InverseVariance'
]:
    input_values[col] = st.number_input(
        label=f"{col}（标准化特征）",
        min_value=float(stats.loc['min', col]),
        max_value=float(stats.loc['max', col]),
        value=float(stats.loc['median', col]),
        step=0.01,
        format="%.4f"
    )

st.markdown("---")
st.markdown("**（2）临床特征**")

# Alcohol：0/1
alcohol_value = st.selectbox(
    "Alcohol（饮酒史：0=无 / 1=有）",
    options=[0, 1],
    format_func=lambda x: "0 = 无饮酒史" if x == 0 else "1 = 有饮酒史"
)
input_values['Alcohol'] = alcohol_value

# AFP
input_values['AFP'] = st.number_input(
    "AFP（ng/mL）",
    min_value=float(stats.loc['min', 'AFP']),
    max_value=float(stats.loc['max', 'AFP']),
    value=float(stats.loc['median', 'AFP']),
    step=1.0,
)

# ALT
input_values['ALT'] = st.number_input(
    "ALT（U/L）",
    min_value=float(stats.loc['min', 'ALT']),
    max_value=float(stats.loc['max', 'ALT']),
    value=float(stats.loc['median', 'ALT']),
    step=1.0,
)

# AST
input_values['AST'] = st.number_input(
    "AST（U/L）",
    min_value=float(stats.loc['min', 'AST']),
    max_value=float(stats.loc['max', 'AST']),
    value=float(stats.loc['median', 'AST']),
    step=1.0,
)

# 将输入整理成模型需要的特征顺序
feature_values = [input_values[col] for col in feature_names]
features_array = np.array([feature_values])
features_df = pd.DataFrame(features_array, columns=feature_names)

st.markdown("---")
st.subheader("2. 模型预测结果")

if st.button("点击进行预测"):
    # -----------------------------
    # 2.1 预测类别 & 概率
    # -----------------------------
    predicted_class = int(model.predict(features_df)[0])
    predicted_proba = model.predict_proba(features_df)[0]  # [p0, p1]

    # 假定 classes_ = [0,1] 且 index 1 为阳性（良好疗效，如 CR/PR）
    prob_non_response = float(predicted_proba[0])  # group = 0
    prob_response = float(predicted_proba[1])      # group = 1

    st.write(f"**预测类别 (group)：{predicted_class}**")
    st.write(f"- 预测为 **0**：多为“非应答/疗效不佳”（如 SD/PD）")
    st.write(f"- 预测为 **1**：多为“应答/疗效较好”（如 CR/PR）")

    st.write("**预测概率（模型输出）**")
    st.write(f"- P(group = 0，非应答)：{prob_non_response * 100:.1f}%")
    st.write(f"- P(group = 1，应答)：{prob_response * 100:.1f}%")

    # -----------------------------
    # 2.2 根据预测类别给出文字解释
    # -----------------------------
    if predicted_class == 1:
        st.success(
            f"模型提示该患者**达到客观缓解（CR/PR）的概率较高**，"
            f"预测 P(应答) ≈ {prob_response * 100:.1f}%。\n\n"
            "这一结果仅基于本研究模型和当前输入特征，不能单独作为治疗决策依据。"
            "建议结合影像学、实验室检查、病理及多学科团队（MDT）评估综合判断。"
        )
    else:
        st.warning(
            f"模型提示该患者**达到客观缓解（CR/PR）的概率相对较低**，"
            f"预测 P(应答) ≈ {prob_response * 100:.1f}%。\n\n"
            "这提示可能存在疗效欠佳或疾病进展的风险，但模型预测存在不确定性。"
            "临床上仍需结合病情、随访影像和主治医生判断，必要时可考虑优化治疗策略。"
        )

    # -----------------------------
    # 2.3 计算 SHAP 值并绘制 force plot
    # -----------------------------
    st.markdown("---")
    st.subheader("3. 模型可解释性（SHAP 力图）")

    with st.spinner("正在计算 SHAP 值并生成图像（可能需要数秒）..."):
        # 对当前样本计算阳性概率的 SHAP 值
        shap_values = explainer.shap_values(features_df)  # shape: (1, n_features)
        shap_values_sample = shap_values[0]

        # 使用 matplotlib 输出 force plot 到图片
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values_sample,
            features_df,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_force_plot_hcc.png", bbox_inches='tight', dpi=300)
        plt.close()

    st.image("shap_force_plot_hcc.png", caption="当前患者的 SHAP 力图（特征对应答概率的贡献）")
