import pickle
import streamlit as st

pickle_in_lr = open('logisticRegr.pkl', 'rb')
classifier_lr = pickle.load(pickle_in_lr)

pickle_in_nb= open('naive bayes.pkl', 'rb')
classifier_nb = pickle.load(pickle_in_nb)

pickle_in_svm = open('svm.pkl', 'rb')
classifier_svm = pickle.load(pickle_in_svm)

pickle_in_rf = open('random forest.pkl', 'rb')
classifier_rf = pickle.load(pickle_in_rf)

st.sidebar.header('Diabetes Prediction (only for people above 21 years of age)')
select = st.sidebar.selectbox('Select Form', ['Form 1','Form 2','Form 3','Form 4'], key='1')

if select=='Form 1': #naive bayes
    st.title('Diabetes Prediction - NAIVE BAYES MODEL')
    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:")
    glucose = st.number_input("Glucose level :")
    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    skin = st.number_input("skin fold thickness (mm):")
    insulin = st.number_input("Insulin level(mu U/ml):")
    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    dpf = st.number_input("Diabetes Pedigree Function:")
    age = st.number_input("Age:")
    submit = st.button('Predict_nb')
    if submit:
         prediction = classifier_nb.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
         if prediction == 0:
            st.write('Congratulation',name,'You are not diabetic')
         else:
            st.write(name," we are really sorry to say but it seems like you are Diabetic.")


if select=='Form 2': #logistic regression
    st.title('Diabetes Prediction - LOGISTIC REGRESSION MODEL')
    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:")
    glucose = st.number_input("Glucose level :")
    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    skin = st.number_input("skin fold thickness (mm):")
    insulin = st.number_input("Insulin level(mu U/ml):")
    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    dpf = st.number_input("Diabetes Pedigree Function:")
    age = st.number_input("Age:")
    submit = st.button('Predict_lr')
    if submit:
         prediction = classifier_lr.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
         if prediction == 0:
            st.write('Congratulation',name,'You are not diabetic')
         else:
            st.write(name," we are really sorry to say but it seems like you are Diabetic.")


if select=='Form 3': #svm
    st.title('Diabetes Prediction - SVM MODEL')
    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:")
    glucose = st.number_input("Glucose level :")
    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    skin = st.number_input("skin fold thickness (mm):")
    insulin = st.number_input("Insulin level(mu U/ml):")
    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    dpf = st.number_input("Diabetes Pedigree Function:")
    age = st.number_input("Age:")
    submit = st.button('Predict_svm')
    if submit:
         prediction = classifier_svm.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
         if prediction == 0:
            st.write('Congratulation',name,'You are not diabetic')
         else:
            st.write(name," we are really sorry to say but it seems like you are Diabetic.")


if select=='Form 4': #random forest
    st.title('Diabetes Prediction - RANDOM FOREST MODEL')
    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:")
    glucose = st.number_input("Glucose level :")
    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    skin = st.number_input("skin fold thickness (mm):")
    insulin = st.number_input("Insulin level(mu U/ml):")
    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    dpf = st.number_input("Diabetes Pedigree Function:")
    age = st.number_input("Age:")
    submit = st.button('Predict_rf')
    if submit:
         prediction = classifier_rf.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
         if prediction == 0:
            st.write('Congratulation',name,'You are not diabetic')
         else:
            st.write(name," we are really sorry to say but it seems like you are Diabetic.")