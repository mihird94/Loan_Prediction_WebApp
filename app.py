import pandas as pd 
import streamlit as st
import joblib
from xgboost import XGBClassifier
import pickle


page_bg_img = '''
<style>
body {
background-image: url("https://coolbackgrounds.io/images/backgrounds/index/aqua-d9b59c89.png");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
#model=joblib.load('dphi3_xgb.pkl')
model=pickle.load(open(r'dphi3_xgb_pickle.pkl','rb'))
#dummycols=joblib.load('dummycols.pkl')
dummycols=pickle.load(open(r'dphi3_dummy_cols.pkl','rb'))

st.markdown("<h1 style='text-align: center; color: black;'>Loan Prophet</h1>", unsafe_allow_html=True)

st.markdown('<p><center>Please enter the details below to check if your loan would be approved<center></p>', unsafe_allow_html=True)
st.markdown('<p><center>This App runs a XGboost model behind the scenes to get the predictions<p><center>', unsafe_allow_html=True)




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

if st.button('Predict'):
    pred=model.predict(test_df_enc)
    label=int(pred)
    st.markdown("<h2 style='color:black;'>The Loan Status is  : %s</h2>" % label_dict[label], unsafe_allow_html=True)



st.markdown('<p><center>(NOTE : Credit history is the most sgnificant feature, please toggle it to see instantaneous changes in predicted outcomes )<p><center>', unsafe_allow_html=True)