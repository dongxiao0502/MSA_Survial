import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
# 加载模型
st.title('Survival Prediction for MSA Patients')
# 设置页面配置
model = joblib.load('modelapplication/rsf_model.pkl')
feature_names = joblib.load("feature_names.pkl")
# 训练集（用于 SHAP 背景）
file_path = r'C:\\Users\\dong\\Desktop\\train111.csv'
X_train = pd.read_csv(file_path)
X_train = X_train.drop(columns=["time", "event"])
background_data = X_train.sample(n=min(50, len(X_train)), random_state=42)

# 创建左右布局：左边用于输入，右边用于输出
col1, col2 = st.columns([5, 20])  # 左边占 1 份，右边占 2 份

# 左侧：输入部分
with st.sidebar:
    st.header('Input Features')
    # 选择框：0 或 1
    FrequentFalls = st.selectbox('Frequent Falls in early 3 years:', ['Yes', 'No'])
    PathologicalSigns = st.selectbox('Pathological Signs :', ['Yes', 'No'])
    Bradykinesia = st.selectbox('Bradykinesia :', ['Yes', 'No'])
    Rigidity = st.selectbox('Rigidity:', ['Yes', 'No'])
    RestTremor = st.selectbox('Rest Tremor :', ['Yes', 'No'])
    NeurogenicOH = st.selectbox('Neurogenic OH :', ['Yes', 'No'])
    UrinaryUrgeIncontinence = st.selectbox('Urinary Urge Incontinence :', ['Yes', 'No'])
    Constipation = st.selectbox('Constipation :', ['Yes', 'No'])

    # 旋钮：数值输入
    ResidualUrine = st.slider('Residual Urine (mL):', min_value=0.0, value=500.0, step=1.0)
    Neutrophils = st.slider('Neutrophils:', min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    Lymphocytes = st.slider('Lymphocytes:', min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    UricAcid = st.slider('Uric Acid (mg/dL):', min_value=0.0, max_value=800.0, value=6.0, step=0.1)
    HCY= st.slider('Homocysteine (µmol/L):', min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    VB9 = st.slider('Vitamin B9 (ng/mL):', min_value=0.0, max_value=20.0, value=20.0, step=0.1)
    Age = st.slider('Age:', min_value=20, max_value=100, value=50, step=1)

    # 计算派生特征
    Ascore = (ResidualUrine > 50) + (UrinaryUrgeIncontinence == 'Yes') + (NeurogenicOH == 'Yes') + (Constipation == 'Yes')
    Pscore =Pscore = (Bradykinesia == 'Yes') + (Rigidity == 'Yes') + (RestTremor == 'Yes')
    NLR = Neutrophils / Lymphocytes if Lymphocytes != 0 else 0

    # 构造输入数据
    input_data = {
        'Age': Age,
        'ResidualUrine': ResidualUrine,
        'UricAcid': UricAcid,
        'HCY': HCY,
        'VB9': VB9,
        'FrequentFalls': 1 if FrequentFalls == 'Yes' else 0,
        'PathologicalSigns': 1 if PathologicalSigns == 'Yes' else 0,
        'NeurogenicOH': 1 if NeurogenicOH == 'Yes' else 0,
        'Ascore': Ascore,
        'Pscore': Pscore,
        'NLR': NLR
    }
    X_new = pd.DataFrame([input_data]) 

    # 构造模型输入格式的 DataFrame
    X_new = X_new[feature_names]
    submit_button = st.button(label='Submit')
# 右侧：输出部分
if submit_button:
    st.header('Prediction Results')
    # 创建四个折叠部分
    with st.expander("Predicted Survival Curve"):
        # 进行预测（假设 RSF 模型有 predict_survival_function 方法）
        surv_funcs = model.predict_survival_function(X_new)
        fn = surv_funcs[0]

        # 绘制预测的生存曲线
        time_points = np.arange(0, 12, 0.1)
        survival_probs = [fn(t) for t in time_points]

        st.line_chart(pd.DataFrame({'Year': time_points, 'Survival Probability': survival_probs}).set_index('Year'))

    with st.expander("Median Survival Time"):
        # 计算中位生存时间
        median_time = None
        for t, s in zip(time_points, survival_probs):
            if s <= 0.5:
                median_time = round(t, 2)
                break

        if median_time:
            st.write(f'Median Survival Time: {median_time} years')
        else:
            st.write('Median survival time could not be determined.')
    with st.expander("Time-point Survival Rate"):
    # 指定感兴趣的时点
    	target_times = [3, 5, 7]  # 年为单位（你也可以换成月）

    # 存储结果
    	for target in target_times:
        # 找到最接近目标时间的索引
        	idx = np.argmin(np.abs(np.array(time_points) - target))
        	surv_prob = survival_probs[idx]
        	st.write(f"{target}-Year Survival Rate: {surv_prob:.2%}")
        

   
    
    with st.expander("Individual SHAP force plot"):
    # 使用 SHAP 解释器
    	   explainer = shap.KernelExplainer(model.predict, background_data)
    	   shap_values = explainer(X_new)
    	   shap_scores = dict(zip(X_new.columns, shap_values[0].values.tolist()))
    shap_df = pd.DataFrame(list(shap_scores.items()), columns=["Feature", "SHAP Value"])
    shap_df = shap_df.sort_values("SHAP Value", ascending=False)

    # 在 Streamlit 上显示每个特征的 SHAP 值
    st.write("SHAP Values for Each Feature:")
    st.dataframe(shap_df)
    # 创建 SHAP 力图并显示
    shap.initjs()
    shap.force_plot(
           explainer.expected_value,  # 使用期望值
           shap_values[0].values,      # 获取 SHAP 值
           X_new.iloc[0],              # 获取当前样本特征
           matplotlib=True             # 使用 matplotlib 绘制
   	    )

    # 显示图表
    st.pyplot(plt) 

   
    with st.expander("Model Explanation"):
    # 你可以展示模型的更多信息，如超参数、模型配置等
        svg_path = r"C:\Users\dong\modelapplication\images\model_architecture.svg"
        with open(svg_path, "r", encoding="utf-8") as file:
            svg_content = file.read()
        st.markdown(f"<div>{svg_content}</div>", unsafe_allow_html=True)
        explanation_text = """
        Each point on the plot represents an individual prediction instance.
        Data points on the right indicate positive contributions to death, while the left represent negative contributions.

        Features are arranged from top to bottom based on the magnitude of their mean absolute SHAP values,indicating their importance in the model's predictions.
        """
    
        st.markdown(explanation_text)  # 在expander中显示解释性文本