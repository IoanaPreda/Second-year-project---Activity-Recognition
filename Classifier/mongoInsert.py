import sys
from pymongo import MongoClient
from datetime import datetime
client = MongoClient('mongodb+srv://user:test@data.vqafg.mongodb.net/Data?retryWrites=true&w=majority')
db = client.Data


def insertData(yhat,X_test,task_id,i,dfTaskRaw):
    currentTime = datetime.utcnow()
    doc = {
        'Task': yhat,
        'Time': currentTime,
        'Task_properties':{
            "JM":{
                'Acc_X' : X_test.iloc[i]['Acc_X-71-JM'],
                'Acc_Y' : X_test.iloc[i]['Acc_Y-71-JM'],
                'Acc_Z' : X_test.iloc[i]['Acc_Z-71-JM'],
                'Gyr_X' : X_test.iloc[i]['Gyr_X-71-JM'],
                'Gyr_Y' : X_test.iloc[i]['Gyr_Y-71-JM'],
                'Gyr_Z' : X_test.iloc[i]['Gyr_Z-71-JM']
            },
            "PN":{
                'Acc_X' : X_test.iloc[i]['Acc_X-71-PN'],
                'Acc_Y' : X_test.iloc[i]['Acc_Y-71-PN'],
                'Acc_Z' : X_test.iloc[i]['Acc_Z-71-PN'],
                'Gyr_X' : X_test.iloc[i]['Gyr_X-71-PN'],
                'Gyr_Y' : X_test.iloc[i]['Gyr_Y-71-PN'],
                'Gyr_Z' : X_test.iloc[i]['Gyr_Z-71-PN']
            },
            "Entropy":{
                'Acc_X' : X_test.iloc[i]['Acc_X-71-EN'],
                'Acc_Y' : X_test.iloc[i]['Acc_Y-71-EN'],
                'Acc_Z' : X_test.iloc[i]['Acc_Z-71-EN'],
                'Gyr_X' : X_test.iloc[i]['Gyr_X-71-EN'],
                'Gyr_Y' : X_test.iloc[i]['Gyr_Y-71-EN'],
                'Gyr_Z' : X_test.iloc[i]['Gyr_Z-71-EN']
            },
            "Accleration":{
                "Acc_X" : dfTaskRaw['Acc_X'].values.tolist(),
                "Acc_Y" : dfTaskRaw['Acc_Y'].values.tolist(),
                "Acc_Z" : dfTaskRaw['Acc_Z'].values.tolist()
            },
            "Gyroscope":{
                "Gyr_X" : dfTaskRaw['Gyr_X'].values.tolist(),
                "Gyr_Y" : dfTaskRaw['Gyr_Y'].values.tolist(),
                "Gyr_Z" : dfTaskRaw['Gyr_Z'].values.tolist()
            }
        },
        'task_id':task_id
    }
    result = db.Tasks.insert_one(doc)
    print('Created {0} of yhat as {1}'.format(i,result.inserted_id))
    message = 'insert done'
    return message



# {
# "Task":"Sample",
# "Time":"time",
# "Task_properties":{
# "JM":[0,1],
# "PN":[0,1],
# "Entropy":[0,1],
# "Accleration":{
# "Acc_X":[0,1],
# "Acc_Y":[0,1],
# "Acc_Z":[0,1]
# }
# }
# }


