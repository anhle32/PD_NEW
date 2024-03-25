from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    Lasso,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report,
    recall_score,
    precision_score,
    roc_auc_score,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import ConfusionMatrixDisplay


np.random.seed(0)


df = pd.read_csv('DATASET.csv', encoding='latin-1')

df[[f"X_{i}" for i in range(1, 15)]].describe()

st.title("DỰ BÁO THAM SỐ PD")
st.write("## Dự báo xác suất vỡ nợ của khách hàng_PD")

uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("DATASET.csv", index = False)

X = df.drop(columns=['default','LGD','EAD'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model= LogisticRegression(random_state=42),

model1=[]
Accuracy=[]
Precision=[]


# Train and evaluate each model
model.fit(X_train, y_train)
y_pred_in_sample = model.predict(X_train)
y_pred_prob_in_sample = model.predict_proba(X_train)[:, 1]

    # Make predictions on the out-sample data
y_pred_out_sample = model.predict(X_test)
y_pred_prob_out_sample = model.predict_proba(X_test)[:, 1]

    # Evaluate performance on in-sample data
accuracy_in_sample = accuracy_score(y_train, y_pred_in_sample)
precision_in_sample = precision_score(y_train, y_pred_in_sample)
recall_in_sample = recall_score(y_train, y_pred_in_sample)
f1_in_sample = f1_score(y_train, y_pred_in_sample)
auc_in_sample = roc_auc_score(y_train, y_pred_prob_in_sample)

    # Evaluate performance on out-sample data
accuracy_out_sample = accuracy_score(y_test, y_pred_out_sample)
precision_out_sample = precision_score(y_test, y_pred_out_sample)
recall_out_sample = recall_score(y_test, y_pred_out_sample)
f1_out_sample = f1_score(y_test, y_pred_out_sample)
auc_out_sample = roc_auc_score(y_test, y_pred_prob_out_sample)

    # Append results to the DataFrame
    # results_df = results_df.append({
    #     'Model': model_name,
    #     'Dataset': 'In-Sample',
    #     'Accuracy': accuracy_in_sample,
    #     'Precision': precision_in_sample,
    #     'Recall': recall_in_sample,
    #     'F1 Score': f1_in_sample,
    #     'AUC': auc_in_sample
    # }, ignore_index=True)

model1.append(model_name)
Accuracy.append(accuracy_out_sample)
Precision.append(precision_out_sample)

dt=pd.DataFrame({'Model':model1,'Accuracy':Accuracy,'Precision':Precision})



menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.write("""
    ###### Mô hình được xây dựng để dự báo xác suất vỡ nợ của khách hàng
    """)  
    st.write("""###### Mô hình sử dụng các thuật toán Random Forest, LogisticRegression""")
    st.image("hinh2.jpg")
    st.image("LogReg_1.png")
    st.image("hinh3.png")

elif choice == 'Xây dựng mô hình':
    st.subheader("Xây dựng mô hình")
    st.write("##### 1. Hiển thị dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    
    st.write("##### 2. Trực quan hóa dữ liệu")
    u=st.text_input('Nhập biến muốn vẽ vào đây')
    fig1 = sns.regplot(data=df, x=u, y='y')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    
    st.write("##### 4. Evaluation")
    st.write(dt)
    st.dataframe(dt)

    

    

    

    
elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("Sử dụng mô hình để dự báo")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            # st.write(lines.columns)
            flag = True       
    if type=="Input":        
        git = st.number_input('Insert y')
        X_1 = st.number_input('Hệ số biên lợi nhuận gộp')
        X_2 = st.number_input('Hệ số biên lợi nhuận trước thuế')
        X_3 = st.number_input('Tỷ suất sinh lời trước thuế trên tổng tài sản')
        X_4 = st.number_input('Tỷ suất sinh lời trước thuế trên vốn chủ sở hữu')
        X_5 = st.number_input('Tỷ số nợ trên tài sản')
        X_6 = st.number_input('Tỷ số nợ trên vốn chủ sở hữu')
        X_7 = st.number_input('Khả năng thanh toán hiện hành')
        X_8 = st.number_input('Khả năng thanh toán nhanh')
        X_9 = st.number_input('Hệ số khả năng trả lãi')
        X_10 = st.number_input('Hệ số khả năng trả nợ gốc')
        X_11 = st.number_input('Hệ số khả năng tạo tiền trên vốn chủ sở hữu')
        X_12 = st.number_input('Vòng quay hàng tồn kho')
        X_13 = st.number_input('Kỳ thu tiền bình quân')
        X_14 = st.number_input('Hiệu suất sử dụng tài sản')




        lines={'y':[git],'X_1':[X_1],'X_2':[X_2],'X_3':[X_3],'X_4':[X_4],'X_5':[X_5],'X_6':[X_6],'X_7':[X_7],'X_8':[X_8],'X_9':[X_9],'X_10':[X_10],'X_11':[X_11],'X_12':[X_12],'X_13':[X_13],'X_14':[X_14]}
        lines=pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            X_1 = lines.drop(columns=['y'])   
            y_pred_new = model.predict(X_1)
            pd=model.predict_proba(X_1)
            st.code("giá trị dự báo: " + str(y_pred_new))
            st.code("PD là: " + str(pd))
