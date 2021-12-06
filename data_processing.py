import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pickle
import math

SESSION_LENGTH = 30 * 60 #30 minutes

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5
MAX_SESSION_LENGTH = 30

#slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
DAYS_TEST = 1

OUTPUT_TYPE = 'train_processed.pkl'

"""

View : 1
addtocart : 2
transaction : 3

"""

"""
Feature consideration
1. Command line argument for the following:
    a. Train/Test split percentage
    b. Validation set included
    3. Debug file enable
2. Optimization needed for max_len (way too much overhead)
3. Possible optimization for data_processing
    a. Why reading sessionId twice? Can it be read globaly?
    b. Moving pandas read outside of this function?
    c. Consider converting events to integers in filter_data or load_data
    


"""


def filter_data( data, min_item_support, min_session_length, max_session_length ) : 
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths<= max_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data

def load_data( ) : 
    
    # load csv
    data = pd.read_csv( 'events'+'.csv', sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int32, 2:str, 3:np.int32})
    # print(type(data))
    
    # specify header names
    data.columns = ['Time','UserId','Type','ItemId']
    item2id_out(data)
    
    data['Time'] = (data.Time / 1000).astype( int )
    # data.sort_values( ['UserId','Time'], ascending=True, inplace=True )
    
    # sessionize    
    data['TimeTmp'] = pd.to_datetime(data.Time, unit='s')
    
    data.sort_values( ['UserId','TimeTmp'], ascending=True, inplace=True )
    
    # check for userdiff before commiting to the current session
    data['UserShift'] = data['UserId'].shift(1)
    data['TimeShift'] = data['TimeTmp'].shift(1)
    data['UserDiff'] = (data['UserId'] - data['UserShift']).abs()
    data['TimeDiff'] = (data['TimeTmp'] - data['TimeShift']).dt.total_seconds().abs()
    
    data['SessionIdTmp'] = 1
    data.iloc[0,9] = 0  # make first row 0
    data.loc[(data.UserDiff == 0) & (data.TimeDiff < SESSION_LENGTH), 'SessionIdTmp'] = 0
    
    data['SessionId'] = data['SessionIdTmp'].cumsum(skipna=False)  # Return cumulative sum over a DataFrame or Series axis.
                                                                     # To include NA values in the operation, use skipna=False
    del data['SessionIdTmp'], data['TimeShift'], data['UserDiff'], data['TimeDiff']
    
    data.sort_values( ['SessionId','TimeTmp'], ascending=True, inplace=True )
    
    #data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    #data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )

    data = filter_data(data, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH)
    data.to_csv("data_output_tmp.csv", index = False)
    return data
    
    
# This function reads from events.csv dataframe and outputs 
# all unique itemid in the events.csv file with a 
# index attached to it. Out file is entity2id
# input: df - dataframe of events.csv
# output: none
def item2id_out(df):
    indexid = [] # list to store index
    df_item = df['ItemId'].unique() # extract unique item id
    df_item = np.insert(df_item, 0, 0) # the very first index contains no item
    # iterate through df_item (list)
    # enumerate will return (index, item)
    # no need to store item as it is already in df_item
    for index, item in enumerate(df_item):
        indexid.append(index)

    # createt new data frame with two column:
    # itemid and index
    df_item = pd.DataFrame({"itemid": df_item,
                            "index" : indexid})

    # write dataframe to entity2id file, no indexing needed
    df_item.to_csv("MKM-SR/data/demo/no_new_item/entity2id", sep = '\t', index = False)  

def data_processing(df,max_seq_len, is_train):
    ''' Need to extract the four types of data needed by MKM-SR
        data_paddings, data_operation_paddings, data_masks, data_targets
        Pseudo code plan:
        1. In the dataframe, extract the unique sequenceId
        2. For each unique sequence id, extract all operation events - for data_operation_paddings
        3. For each unique sequence id, extract items associated with each operation events - for data_paddings
        4. For each unique sequence id, extract the last operation events item - for data_targets
        5. Add the unique sequence id to each data extraction (for future sanity check)
        6. If needed, process each data into picke stream
    '''

    sessionId_list = df["SessionId"].unique() # list of unique sessionId, step 1
    itemid_list = pd.read_csv("MKM-SR/data/demo/no_new_item/entity2id", sep = "\t") 
    itemid_list = itemid_list["itemid"] # need to get entire itemid_list
    data_targets = [] # list to store the target for each session
    data_paddings = []
    data_operation_paddings = []
    data_masks = []
    #max_seq_len = max_seq_len
    # step 2 - 6
    for index, sessionId in enumerate(sessionId_list):
       # if(index%1000 == 0):
        #    print(index)
        df_seq = df.loc[df['SessionId'] == sessionId]   # extract all sequences associated with this session
        #print(df_seq)
        seq_events_list = list(df_seq['Type'])  # extract all events associated with this given sequences, step 2
        # convert type string to integers
        for index, item in enumerate(seq_events_list):  
            if(item == 'view'):
                seq_events_list[index] = 1
            elif(item == 'addtocart'):
                seq_events_list[index] = 2
            elif(item == "transaction"):
                seq_events_list[index] = 3
        seq_items_list = list(df_seq['ItemId'])  # extract all itemid associated with this given sequences, step 3
        seq_items_list = [np.where(itemid_list == item)[0][0] for item in seq_items_list]
        seq_mask_list = [1 for item in seq_items_list]
        df_seq_last = df_seq.iloc[-1:]  # locate the final operation events associated with this sequenceId

        # store the itemid associated with this event, step 4 and 5
        target = np.where(itemid_list == (int(df_seq_last['ItemId'])))[0][0]
        target_tuple = (int(df_seq_last['ItemId']), int(df_seq_last['SessionId'])) 
        data_targets.append(target)    # step 6
        data_paddings.append(seq_items_list[:-1])
        data_operation_paddings.append(seq_events_list[:-1])
        data_masks.append(seq_mask_list[:-1])

    # pad all data_paddings and data_operation_paddings to the max_sequence_len
    for seq_item, seq_event,seq_mask, index in zip(data_paddings, data_operation_paddings, data_masks, range(len(data_paddings))):
        if(len(seq_item) < max_seq_len):
            seq_item = seq_item + [0 for i in range(max_seq_len - len(seq_item))]
            seq_event = seq_event + [0 for j in range(max_seq_len - len(seq_event))]
            seq_mask = seq_mask + [0 for k in range(max_seq_len - len(seq_mask))]
            data_paddings[index] = seq_item
            data_operation_paddings[index] = seq_event
            data_masks[index] = seq_mask
    train_process_data = [data_paddings, data_operation_paddings, data_masks, data_targets]
    if(is_train):
        with open("MKM-SR/data/demo/no_new_item/MKM_SR/train_processed.pkl", 'wb') as file_1:
            pickle.dump(train_process_data, file_1, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        with open("MKM-SR/data/demo/no_new_item/MKM_SR/test_processed.pkl", 'wb') as file_1:
            pickle.dump(train_process_data, file_1, protocol = pickle.HIGHEST_PROTOCOL)

# find the maximum session length of the dataframe
def find_max_len(df):
    max_seq_len = 0
    unique_sessionId = df["SessionId"].unique() # extract unique session ID

    # iterate over unique session ID
    for sessionId in unique_sessionId:
        df_seq = df.loc[df["SessionId"] == sessionId]
        seq_list = list(df_seq["SessionId"])
        max_seq_len = max(max_seq_len, len(seq_list))

    return max_seq_len

def split_train_test(data):
    """
    This function is used to split the given data frame into training data frame
    and testing data frame into
    
    @input: data
    @output: data_train, data_test
    """
    # extract all unique sessionId
    unique_sessionId = data["SessionId"].unique()
    demo_size = len(unique_sessionId)-1   # for demo use 50000 total session
    cutoff_thre = math.ceil(demo_size * 0.8)    # 80 percent training set, 20 percent testing set
    cutoff_sessionId = unique_sessionId[cutoff_thre]    # locate the sessionId for the cutoff threshold
    final_sessionId = unique_sessionId[demo_size]
    train_test_threshold = data[data["SessionId"] == cutoff_sessionId].index[-1] + 1   # find all places where it uses this sessionId， grab the final one
    final_cutoff = data[data["SessionId"] == final_sessionId].index[-1] + 1
    data_demo = data[0:final_cutoff]
    item2id_out(data_demo)  # extract all unique ItemID for this demo data
    data_train = data[0:train_test_threshold]   # split according to index threshold
    data_test = data[train_test_threshold:final_cutoff]
    max_seq_len = find_max_len(data[0:final_cutoff])
    return data_train, data_test, max_seq_len

def main():
    # read events.csv and open a dataframe handle
   # df = pd.read_csv("./events.csv")
    # write unique itemid in events.csv file to entity2id with indexing
   # item2id_out(df)

  #  data_processing(df)
  data = load_data()
  data_split = pd.read_csv("data_output_tmp.csv")
  data_train, data_test, max_seq_len = split_train_test(data_split)
  
  print("split done")
  data_processing(data_train, max_seq_len, True)
  data_processing(data_test, max_seq_len, False)
    

if __name__ == '__main__':
    main()