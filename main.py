import numpy as np
import pickle 
import pandas as pd
import streamlit as st
from file_operations import file_methods
from preprocessing.preprocessing import preprocessing
from preprocessing.clustering import KMeansClustering
from application_logger import logger
file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
log_writer = logger.app_logger()

def predict(list1):
    try:
        columns=['Airline','Date_of_Journey','Source','Destination','Total_Stops','Duration','Dep_Time','Arrival_Time']
        data=pd.DataFrame(list1,columns=columns)
        preprocessor=preprocessing(file_object,log_writer)
        data=preprocessor.drop_missing_values(data)
        data=preprocessor.change_format('Date_of_Journey',data)
        data=preprocessor.process_date(data)
        data=preprocessor.preprocess_duration(data)
        data=preprocessor.preprocess_departure(data)
        if len(list1[-1][-1])>5:
            data=preprocessor.preprocess_arrival(data)
        else:
            data=preprocessor.preprocess_arrival_pred(data)
        data=preprocessor.encode_pred(data)
        file_loader=file_methods.File_Operation(file_object,log_writer)
        kmeans=file_loader.load_model('KMeans')
        clusters=kmeans.predict(data)#drops the first column for cluster prediction
        data['clusters']=clusters
        clusters=data['clusters'].unique()
        result=[] # initialize balnk list for storing predicitons
        for i in clusters:
            cluster_data= data[data['clusters']==i]
            cluster_data = cluster_data.drop(['clusters'],axis=1)
            model_name = file_loader.find_correct_model_file(i)
            model = file_loader.load_model(model_name)
            for val in (model.predict(cluster_data)):
                result.append(val)
            result= result[0]
            log_writer.log(file_object,'End of Prediction')
            return result
    except Exception as ex:
            log_writer.log(file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
def main():
    st.title("Flight Fare Prediction")
    html_temp="""
    <div style="background-color:lightblue;padding:10px">
    <h2 style="color:white;text-align:center;">Flight Fare Prediction</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Airlines=st.text_input("Airline Name","Type here")
    Date_of_Jounrey=st.text_input("Date","Type here")
    Source=st.text_input("Source","Type here")
    Destination=st.text_input("Destination","Type here")
    Total_Stops=st.text_input("Total_Stops","Type here")
    Duration=st.text_input("Duration","Type Here")
    Dep_time=st.text_input("Departure Time","Type here")
    Arrival_Time=st.text_input("Arrival Time","Type here")

    result=""
    if st.button("Predict"):
        result=np.round(predict([[Airlines,Date_of_Jounrey,Source,Destination,Total_Stops,Duration,Dep_time,Arrival_Time]]))
    st.success('Verdict: {}'.format(result))
if __name__=="__main__":
    main()

