
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#config for plotly to not show the mode bar (zoom etc...)
config = {'displayModeBar': False}

#---------LOAD DATA AND MODEL------------------------------------

#load needed data
with open("ressources/names.txt","r") as f:
  names = f.readlines()
names = [name.replace('\n',"") for name in names]
#capitalize names
names = [name.capitalize() for name in names]

features = np.load("ressources/features.npy",allow_pickle=True)
#doing some renaming on features names
features = [[feat.capitalize() if "twi" not in feat else "Twi" for feat in feature] for feature in features]

names_train, names_test, features_train, features_test = train_test_split(names,features,random_state = 50, test_size=0.1,)

namesVsFeatures = dict(zip(names,features))

with open('ressources/tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)
with open("ressources/classes.txt","r") as f:
  classes = f.readlines()
classes = [cl.replace('\n',"") for cl in classes]
#doing some renaming on classes names
classes = [classe.capitalize() if "twi" not in classe else "Twi" for classe in classes]

repr_train = np.load("ressources/repr_train.npy")
repr_test = np.load("ressources/repr_test.npy")
namesFeaturesDetector = load_model("ressources/nameFeaturesDetector.h5")
representation = load_model("ressources/nameRepresentation.h5")

#this function is also available in utils.py
def predictionPipeline(name,classes,tokenizer,model,representation,inputLen):
  '''
  A function taking as input a name and building blocks of a prediction pipeline, 
  and returning as a dictionnary predicted features: locality, firstVsLastName, femaleVsMale,
  as well as an embedding vector.
  Parameters:
    name (str): a name to be processed.
    classes (list(str)): a list of the names of the classes predicted by the model.
    tokenizer (keras tokenizer): a tokenizer with a vocabulary, trained on the same train set as the model.
    model (keras model): a trained model for names features inference.
    representation (keras model): a model output embeddings for the name.
    inputLen (int): the model's inputs length.
  Returns:
    namesFeaturesDict (dict): a dictionnary with predicted features: locality, firstVsLastName, femaleVsMale
  '''
  name = name.lower()


  # tokenize names with the tokenizer
  sequence = tokenizer.texts_to_sequences([name])
  sequence = pad_sequences(sequences=sequence, maxlen=inputLen, padding='post')

  predictions = model.predict(sequence).ravel()

  nameFeaturesDict = dict()

  nameFeaturesDict["firstVsLastName"] = dict(firstName=predictions[1],lastName=predictions[3])
  nameFeaturesDict["femaleVsMale"] = dict(female=predictions[0],male=predictions[4])  
  nameFeaturesDict["locality"] = dict(zip(classes[5:],predictions[5:]))

  #representation
  repr = representation.predict(x=sequence)
  norm = np.linalg.norm(repr,axis=1).reshape(-1,1)
  repr = repr/norm

  nameFeaturesDict["representation"] = repr

  return nameFeaturesDict

#-----------DEFINE DASH APP-------------------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go


app = dash.Dash(__name__)

#for Heroku deployment
server = app.server

#---Define layout-----------------------------------

graphLayout = go.Layout(template="plotly_dark",xaxis=dict(tickmode="linear",showgrid=False,visible=False),
                       paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
#look at the ids of the Div to understand their purposes
app.layout = html.Div(
    [
        html.Title("NameFeaturesDetector"),
        html.Div(
            [
                html.P("Names Features Detector",style = {"padding-left":"2%","font-size":"18px"})
            ],
            id = "header"
        ),
        html.Div(
            [
                html.Div(
                    ["Enter a Name: ",
                     dcc.Input(value="John",type="text",id="name"),
                     html.Button(id='run-button', n_clicks=0, children='Run',style={"margin":"1%"}),
                     html.Button(id='pickrandom-button', n_clicks=0, children='Pick Random')
                    ],
                    style = {"padding-left":"5%","font-size":"14px"}
                )
            ],
            id = "parameters"
        ),
        html.Div(
            [
                
            ],
            id = "text-output"
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="firstVsLast",config=config,figure=go.Figure(layout=graphLayout)),
                        dcc.Graph(id="femaleVsMale",config=config,figure=go.Figure(layout=graphLayout)),
                        dcc.Graph(id="similarity",config=config,figure=go.Figure(layout=graphLayout)),
                        dcc.Graph(id="locality",config=config,figure=go.Figure(layout=graphLayout)),
                        dcc.Graph(id="locSimilar",config=config,figure=go.Figure(layout=graphLayout))
                    ],
                    className = "graphs-container",
                )
            ],
            id= "graphs-output"
        )
    ],
    className = "main-container",

)

#------Define callbacks----------------------

#this callback gets as input the run button event or the pick random button event
#it will then ouput the locality of the name, its gender, if it is a first or last name
#and similar names based on the similarity measure defined via last layer embeddings.

@app.callback(
    [Output("firstVsLast","figure"),
     Output("femaleVsMale","figure"),
     Output("similarity","figure"),
     Output("locality","figure"),
     Output("locSimilar","figure"),
     Output("name","value"),
     Output("text-output","children")],
    [Input("run-button","n_clicks"),
     Input("pickrandom-button","n_clicks")],
    [State("name","value")]
)


def update_graphs(n_clicks1,n_clicks2,name):

    #this is to understand which button has been clicked
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    #if the "pick random" button was clicked then pick a name at random in the test set
    if button_id == "pickrandom-button":
      rndIndex = np.random.randint(0,len(names_test))
      name = names_test[rndIndex]
    
    name = name.lower()

    #get the data from the prediction pipeline
    data = predictionPipeline(name,classes,tk,namesFeaturesDetector,representation,inputLen=12)
    
    firstVsLast = pd.Series(data["firstVsLastName"]).sort_values(ascending=True)
    femaleVsMale = pd.Series(data["femaleVsMale"]).sort_values(ascending=True)
    locality = pd.Series(data["locality"]).sort_values(ascending=True).iloc[-10:]

    #compute cosine similarity from name embeddings, and extract the most similar 10 names
    similarity_train = np.dot(data["representation"],repr_train.T)
    similarity_train = pd.Series(data=similarity_train.ravel(),
                               index=names_train).sort_values(ascending=False).iloc[:20].iloc[::-1]

    #text for similarity 
    text = [namesVsFeatures[name] for name in similarity_train.index]

    #locality for similar names
    localities = [locality for locality in [feature for features in text for feature in features if feature not in ["Last","Firstname","Lastname","Male","Female"]]]
    locCounter = Counter(localities)
    locSimilar = pd.Series(dict(locCounter.most_common(10))).iloc[::-1]
    
    colorscale = "Blugrn"
    #figures
    localityBar = go.Bar(y=locality.index,x=locality.values,text=locality.values.round(2)*100,textposition='outside',textfont_color="white",
                         orientation="h",showlegend=False,marker={'color': locality.values,'colorscale': colorscale}
                         )

    femaleVsMaleBar = go.Bar(y=femaleVsMale.index,x=femaleVsMale.values,text=femaleVsMale.values.round(2)*100,textposition='outside',textfont_color="white",
                             orientation="h",showlegend=False,marker={'color': femaleVsMale.values,'colorscale': colorscale}, 
                             )

    firstVsLastBar = go.Bar(y=firstVsLast.index,x=firstVsLast.values,text=firstVsLast.values.round(2)*100,textposition='outside',textfont_color="white",
                            orientation="h",showlegend=False,marker={'color': firstVsLast.values,'colorscale': colorscale}
                           )

    similarityBar = go.Bar(y=similarity_train.index,x=similarity_train.values,
                           orientation="h",showlegend=False,text=text,marker={'color': similarity_train.values,'colorscale': colorscale}
                           )

    locSimilarBar = go.Bar(y=locSimilar.index,x=locSimilar.values,textposition='outside',
                           orientation="h",showlegend=False,text=locSimilar.values,marker={'color': locSimilar.values,'colorscale': colorscale}
                           )

    fig1 = go.Figure([firstVsLastBar]).update_layout(margin=dict(t=30,b=30))
    fig2 = go.Figure([femaleVsMaleBar]).update_layout(margin=dict(t=30,b=30))
    fig3 = go.Figure([similarityBar]).update_layout(margin=dict(t=30,b=50))
    fig4 = go.Figure([localityBar]).update_layout(margin=dict(t=30,b=10))
    fig5 = go.Figure([locSimilarBar]).update_layout(margin=dict(t=30,b=50))
    
    #titles of the barplots
    titles = ["Is it a first name or a last name ?",
              "Is it a female or male name ?",
              "What are the most similar names in the train set ?",
              "Locality predicted by the model","Locality among most similar names"
             ]
    
    for fig,title in zip([fig1,fig2,fig3,fig4,fig5],titles):
        fig.update_layout(title_text=title,template="plotly_dark",xaxis=dict(tickmode="linear",showgrid=False,visible=False),
                       paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
                       
    #message to output (this could maybe done more nicely)
    
    #capitalize name
    name = name.capitalize()
    mess = str()
    if name in names:
        postmess = ""
        feats = namesVsFeatures[name]
        if "Firstname" in feats:
            postmess += " a firstname"
            if "Lastname" in feats: 
                postmess+= " and a lastname,"
            else: postmess+= " only,"
            
            if "Male" in feats:
                postmess+= " a male name"
                if "Female" in feats: 
                    postmess += " and a female name,"
                else: 
                    postmess += " only,"
                
        else: postmess += " a lastname only,"
        
        locas = [feat for feat in feats if feat not in ["Firstname","Lastname","Male","Female","Last"]]
        
        postmess += " possible localities are  " + ", ".join(locas[:-1]) + ", "*(len(locas) == 2) + locas[-1] + "."
    
    
    if name in names_train: 
        mess = name + " was in the train dataset. It can be " + postmess
    elif name in names_test: 
        mess = name + " was in the test dataset. It can be " + postmess
    else: mess = name + " was not in the train nor in the test dataset."
        
    return fig1,fig2,fig3,fig4,fig5,name,mess

if __name__ == '__main__':
    app.run_server(debug=False)