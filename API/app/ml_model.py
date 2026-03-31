import pickle

with open('./app/moodify_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_prediction(df_features):

    try:
        label = model.predict(df_features)
        return label[0]
    except:
        return None