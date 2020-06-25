import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    st.title("Machine Learning Classifiers Tool.")
    def file_select(folder='./datasets'):
        filelist=os.listdir(folder)
        selectedfile=st.selectbox('select a default file',filelist)
        return os.path.join(folder,selectedfile)


    if st.checkbox('Select dataset from local machine'):
        data=st.file_uploader('Upload Dataset in .CSV',type=['CSV'])
        if data is not None:
            df=pd.read_csv(data)
    else:
        filename=file_select()
        st.info('You selected {}'.format(filename))
        if filename is not None:
            df=pd.read_csv(filename)
            
    #columns set for model building
    collist=df.columns.tolist()
    st.write("Select Columns for Model Building:")
    featlist=st.multiselect("Select:",collist)
    #Train-Test-Split
    from sklearn.model_selection import train_test_split
    X=df[featlist].values
    y=df.iloc[:,-1].values
    testsize=st.slider("Select Test Data Size for splitting:",10,50)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testsize/100,random_state=209)
    st.write("Selected Data:")
    st.write(df[featlist])
    
    #Classifiers
    selected_model=st.selectbox("Select a Classifier:",['LogisticRegression','KNN','NaiveBayes','DecisionTree','SVM','RandomForest'])
    #LR
    if selected_model=='LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        logreg=LogisticRegression()
        logreg.fit(X_train,y_train.ravel())
        pred_logreg=logreg.predict(X_test)
        if st.button("Get Prediction"):
            st.write("Accuracy:",metrics.accuracy_score(y_test,pred_logreg))
            st.write("Recall:",metrics.accuracy_score(y_test,pred_logreg))
            st.write("Precision:",metrics.accuracy_score(y_test,pred_logreg))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_logreg))
            
    #KNN
    if selected_model=='KNN':
        from sklearn.neighbors import KNeighborsClassifier
        scoreknn=[]
        avgknnscore=0
        num=st.slider("Select N Nearest Neighbours:",1,25)
        k_range=list(range(1,num+1))
        for k in range(1,num+1):
            knn=KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train,y_train)
            pred_knn=knn.predict(X_test)
            scoreknn.append(metrics.accuracy_score(y_test,pred_knn))
            avgknnscore+=metrics.accuracy_score(y_test,pred_knn)
        if st.button("Get Prediction"):
            st.write("Used 1 to N Nearest Neighbours.")
            st.write("Average Accuracy:",avgknnscore/num)
            st.write("Recall:",metrics.accuracy_score(y_test,pred_knn))
            st.write("Precision:",metrics.accuracy_score(y_test,pred_knn))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_knn))
            #plot
            plt.plot(k_range, scoreknn)
            plt.xlabel('Value of k for KNN')
            plt.ylabel('Accuracy Score')
            plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
            st.pyplot()
            
    #NB
    if selected_model=='NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        nb=GaussianNB()
        nb.fit(X_train,y_train)
        pred_nb=nb.predict(X_test)
        if st.button("Get Prediction"):
            st.write("Accuracy:",metrics.accuracy_score(y_test,pred_nb))
            st.write("Recall:",metrics.accuracy_score(y_test,pred_nb))
            st.write("Precision:",metrics.accuracy_score(y_test,pred_nb))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_nb))
            
    #DT
    if selected_model=='DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree
        dt=DecisionTreeClassifier()
        dt.fit(X_train,y_train)
        pred_dt=dt.predict(X_test)
        if st.button("Get Prediction"):
            st.write("Accuracy:",metrics.accuracy_score(y_test,pred_dt))
            st.write("Recall:",metrics.accuracy_score(y_test,pred_dt))
            st.write("Precision:",metrics.accuracy_score(y_test,pred_dt))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_dt))
            
    #SVM
    if selected_model=='SVM':
        from sklearn import svm
        svmmodel=svm.SVC()
        svmmodel.fit(X_train,y_train)
        pred_svm=svmmodel.predict(X_test)
        if st.button("Get Prediction"):
            st.write("Accuracy:",metrics.accuracy_score(y_test,pred_svm))
            st.write("Recall:",metrics.accuracy_score(y_test,pred_svm))
            st.write("Precision:",metrics.accuracy_score(y_test,pred_svm))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_svm))
            
    #RF
    if selected_model=='RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        rfc=RandomForestClassifier(random_state=29)
        rfc.fit(X_train,y_train)
        pred_rfc=rfc.predict(X_test)
        if st.button("Get Prediction"):
            st.write("Accuracy:",metrics.accuracy_score(y_test,pred_rfc))
            st.write("Recall:",metrics.accuracy_score(y_test,pred_rfc))
            st.write("Precision:",metrics.accuracy_score(y_test,pred_rfc))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_rfc))
            
    #Footer-info
    if st.button("See who created this!"):
        st.info("Name: K. Vikas Reddy")
        st.info("College: SASTRA Deemed to be University")
        st.info("Gmail: reddyvikas995@gmail.com")
        st.write("You can also see my other projects at:")
        st.write("1. https://edawithease.herokuapp.com to perform Exploratory Data Analysis.")
        st.write("2. http://diabetes-prediction-ml.herokuapp.com to predict diabetic condition.")
        
    st.warning("Please report any bugs and suggestions if any.")
        
    if st.checkbox("About this Project"):
        st.write("Data Science is a rapidly growing field.As Data Science is growing fast, Data Science aspirants should also become smart and productive.")
        st.write("Whenever starting a new project/problem, it is always advisable to first DRY RUN(DIRTY RUN) the problem using simple ML models.")
        st.write("This is the main objective of this project. To help people perform the simple dry-run process.")
        st.write("DRY-RUN is important because it helps in better understanding of the ML process. It helps in answering the following questions:")
        st.write("1. Should I collect more data?")
        st.write("2. Should I reduce the number of features?")
        st.write("3. Should I increase the number of features?")
        st.write("4. Should I perform Feature Scaling?")
        st.write("5. Is the model Overfitting?")
        st.write("6. Is the model Underfitting? and a lot more...")
        st.write("For example, if the model is Underfitting, collecting more(additional) data would not help much.")
        st.write("If you know this before spending valuable time in collecting more data, it would save you the time.")
        st.write("Not just above example, there are a lot more cases where DRY-RUN can help saving time and resourses.")
        st.subheader("Note:")
        st.write("Please use the clean and processed dataset for model building and ensure the last feature is the dependent feature(Output).")
    
    
if __name__=='__main__':
    main()
