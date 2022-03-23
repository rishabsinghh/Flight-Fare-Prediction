from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import preprocessing
from preprocessing.clustering import KMeansClustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logger import logger 
import pandas as pd
try:   
    log_writer = logger.app_logger()
    file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+') 
    log_writer.log(file_object, 'Start of Training')
    data=pd.read_excel('Train.xlsx')
    preprocessor=preprocessing(file_object,log_writer)
    data=preprocessor.remove_columns(['Route','Additional_Info'],data)
    data=preprocessor.drop_missing_values(data)
    data=preprocessor.change_format('Date_of_Journey',data)
    data=preprocessor.process_date(data)
    data=preprocessor.preprocess_duration(data)
    data=preprocessor.preprocess_departure(data)
    data=preprocessor.preprocess_arrival(data)
    data=preprocessor.process_totalstops(data)
    data=preprocessor.encode_categorical_variables(data)
    print(data)
    X,Y=preprocessor.separate_label_feature(data,'Price')
    kmeans=KMeansClustering(file_object,log_writer)
    number_of_clusters=kmeans.elbow_plot(X) 
    # Divide the data into clusters
    X=kmeans.create_clusters(X,number_of_clusters)
    #create a new column in the dataset consisting of the corresponding cluster assignments.
    X['Labels']=Y
    # getting the unique clusters from our dataset
    list_of_clusters=X['Cluster'].unique()
    """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
    for i in list_of_clusters:
        cluster_data=X[X['Cluster']==i] # filter the data for one cluster
       # Prepare the feature and Label columns
        cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
        cluster_label= cluster_data['Labels']
         # splitting the data into training and test set for each cluster one by one
        x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
        model_finder=tuner.Model_Finder(file_object,log_writer) # object initialization
        #getting the best model for each of the clusters
        best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)
        #saving the best model to the directory.
        file_op = file_methods.File_Operation(file_object,log_writer)
        save_model=file_op.save_model(best_model,best_model_name+str(i))
        # logging the successful Training
    log_writer.log(file_object, 'Successful End of Training')
    file_object.close()

except Exception as e:
    log_writer.log(file_object, 'Unsuccessful End of Training')
    file_object.close()
    raise Exception
    
