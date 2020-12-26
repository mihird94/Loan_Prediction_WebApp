import pandas as pd 
import streamlit as st
import joblib
import pickle
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as pl
page_bg_img = '''
<style>
body {
background-image: url("https://coolbackgrounds.io/images/backgrounds/index/aqua-d9b59c89.png");
background-size: cover;
}
</style>
'''
st.set_option('deprecation.showPyplotGlobalUse', False)
X=pickle.load(open(r'dphi3_X_for_shap.pkl','rb'))
st.markdown(page_bg_img, unsafe_allow_html=True)

model=pickle.load(open(r'dphi3_rf_pickle.pkl','rb'))

dummycols=pickle.load(open(r'dphi3_dummy_cols.pkl','rb'))

explainer=pickle.load(open(r'dphi3_rf_shap_explainer.pkl','rb'))

st.markdown("<h1 style='text-align: center; color: black;'>Loan Prophet</h1>", unsafe_allow_html=True)

st.markdown('<p><center>Please enter the details below to check if your loan would be approved<center></p>', unsafe_allow_html=True)
st.markdown('<p><center>This App runs a RandomForest model behind the scenes to get the predictions and SHAP to explain the predictions<p><center>', unsafe_allow_html=True)




gender = st.selectbox(
    'Please select your Gender: ',
  ('Male','Female'))

married = st.selectbox('Are you Married?',('Yes','No'))

dependents = st.selectbox('How many dependents do you have?',('0','1','2','3+'))

education = st.selectbox('Select your graduation status: ',('Graduate','Not Graduate'))

self_emp=st.selectbox('Are you self employed?',('Yes','No'))

property_area = st.selectbox('Select your property area: ',('Semiurban', 'Rural', 'Urban'))

credit_history=st.selectbox('What is your credit history status?',(0.0,1.0))

st.markdown("*Credit History is a record of a borrower's responsible repayment of debts (1- has all debts paid, 0- not paid)*")
appl_income=st.number_input('Please Enter Applicant Income: ',value=5401.189409,step=200.00,min_value=0.0)

coappl_income=st.number_input('Please Enter Co-Applicant Income: ',value=1589.730998,step=200.00,min_value=0.0)

loan_amt=st.number_input('Please enter loan amount: ',value=145.0147368,step=20.00,min_value=0.0)

loan_amt_term=st.number_input('Please enter loan amount term: ',value=341.2970711,step=20.00,min_value=0.0)

data_dict={'Gender':gender, 'Married':married, 'Dependents': dependents, 'Education':education, 'Self_Employed':self_emp,
       'ApplicantIncome':appl_income, 'CoapplicantIncome':coappl_income, 'LoanAmount':loan_amt,
       'Loan_Amount_Term':loan_amt_term, 'Credit_History':credit_history, 'Property_Area':property_area }
test_df=pd.DataFrame(data_dict,index=[0])

test_df_enc=pd.get_dummies(test_df)
test_df_enc=test_df_enc.reindex(columns=dummycols,fill_value=0)


label_dict={0:'Not Approved',1:'Approved!'}

shap_values=explainer.shap_values(test_df_enc)
pred=model.predict(test_df_enc)
label=int(pred)

left, right = st.beta_columns(2)
with left:
    if st.button('Predict'):
        st.markdown("<h2 style='color:black;'>The Loan Status is  : %s</h2>" % label_dict[label], unsafe_allow_html=True)
with right:
    if st.button('Explain'):
        st.markdown("<h2 style='color:black;'>%s</h2>" % label_dict[label], unsafe_allow_html=True)
        st.write('Which features caused this specific prediction? features in red increased the prediction towards approved, in blue decreased them')
        st.write('Please Zoom in to View the SHAP plot!')
        
        fig=shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],test_df_enc.iloc[0,:],matplotlib=True,show=False,figsize=(20,5))
   
        st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)
        pl.clf()
        
        st.write('Detailed Table of the SHAP Values given below: ')
        
        shap_table=pd.DataFrame(shap_values[1],columns=dummycols)
        st.table(shap_table.iloc[0])
if st.button('Show Global Feature Importance'):
    st.write('This is the summary plot based on Shapley values for the training set, The higher the average absolute shap value, more more the feature. In Our case, Credit History is the most important feature picked by the model')
    fig2=shap.summary_plot(X[1], X[0], plot_type='bar',show=False)
    st.pyplot(fig2,bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
        
st.markdown('<p><center>(NOTE : Credit history is the most significant feature, please toggle it to see instantaneous changes in predicted outcomes )<p><center>', unsafe_allow_html=True)
st.markdown('Read More about SHAP here:- [Link to Documentation](https://shap.readthedocs.io/en/latest/)')