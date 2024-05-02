
import pandas as pd
import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
import json

import sys
sys.path.append('data/')

# Jobs Dataset
df = pd.read_csv('data/li_jobs_emb.zip')

# Create a dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL]) #, assets_folder=assets_path

###########
# Access model's container
url = 'http://model_api:8000/single_embeddings'
url = 'http://localhost:8000/single_embeddings'
###########


title = 'Base recommender demo'
logo = 'logo'
color_cards = '#47a8ed'
color_text = '#ed7947'

header = html.Div([
    html.Span(className='helper'),
    html.Div(html.Img(src=f'assets/{logo}.png', className='logo'),
             style={'width': '20%', 'opacity': 0.95, 'align-self': 'center'}),
    html.Div(html.B(title, style={'textAlign': 'center', 'color': color_text, 'font-size': 22}),
             style={'width': '60%', 'text-align': 'center', 'vertical-align': 'middle', 'align-self': 'center'}),
    html.Div(style={'width': '20%', 'opacity': 0.9})
], className='header')


input_groups = dbc.Container([html.Div(
    [html.Br(),
        dbc.InputGroup(
            [dbc.InputGroupText("@"), dbc.Input(placeholder="Username", id="username_input")],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Job Title"),
                dbc.Input(placeholder="Desired position", id="title_input"),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Summary"),
                dbc.Textarea(placeholder="Describe your experience", id="desc_input"),
            ],
            className="mb-3",
        ),
    ]
)])

def get_user_rec(emb, num_rec=10):
    emb = np.asarray(json.loads(emb))

    job_emb = np.asarray(df['emb'].apply(lambda x: json.loads(x)).to_list())
    top_jobs = (-1 * (emb @ job_emb.T)).argsort(axis=1)[:, :num_rec]
    return df.iloc[top_jobs[0]]

def get_jumbotron(title, text, location):
    jt = html.Div([
        html.H4(f"{title}", className="display-7"),
        html.Hr(className="my-2"),
        html.Div(
        html.P(
            f"{text}"
        ), style={'height': '250px', 'overflow':'auto'}
        ),
        html.Hr(className="my-2"),
        html.Div(html.P(f'📍 {location}'), style={'text-align': 'right'}),
        #dbc.Button("Example Button", color="secondary", outline=True),
    ], className="h-100 p-5 bg-light border rounded-3")

    return jt


def get_jumbotron(title, text, location):
    jt = html.Div([
        dbc.CardHeader(f'📍 {location}'),

        dbc.CardBody(
            [
                html.H5(f"{title}", className="card-title"),
                html.P(
                    text,
                    className="card-text",
                ),
            ], style={'height': '250px', 'overflow':'auto'}
        )

        #dbc.Button("Example Button", color="secondary", outline=True),
    ])

    return dbc.Card(jt, color="dark", outline=True, style = {"box-shadow": "0 0 10px rgba(0,0,0,0.5)"})


first_jumbotron = dbc.Col(
    html.Div([],
    ), id="first_jumbotron",
    md=4,
)

second_jumbotron = dbc.Col(
    html.Div([],
    ), id="second_jumbotron",
    md=4,
)

third_jumbotron = dbc.Col(
    html.Div([],
    ), id="third_jumbotron",
    md=4,
)

fourth_jumbotron = dbc.Col(
    html.Div([],
    ), id="fourth_jumbotron",
    md=4,
)
fifth_jumbotron = dbc.Col(
    html.Div([],
    ), id="fifth_jumbotron",
    md=4,
)
sixth_jumbotron = dbc.Col(
    html.Div([],
    ), id="sixth_jumbotron",
    md=4,
)

jumbotron = dbc.Container(
    dcc.Loading(type="circle", children = dbc.Row(
        [first_jumbotron, second_jumbotron, third_jumbotron, fourth_jumbotron, fifth_jumbotron, sixth_jumbotron],
    className="align-items-md-stretch",
), className="h-100", style={'max_height': '60%'}))

jumbotron = dbc.Container(
    dcc.Loading(type="circle", children = [
                        dbc.Row(
                                [first_jumbotron, second_jumbotron, third_jumbotron],
                            className="align-items-md-stretch",
                        ),
                        html.Br(),
                        dbc.Row(
                                [fourth_jumbotron, fifth_jumbotron, sixth_jumbotron],
                            className="align-items-md-stretch",
                        )],
                className="h-100", style={'max_height': '60%'}))



submit_button = dbc.Container(
    dbc.Button("Submit", color="primary", id='update_output', outline=True,
               style={'width': '70%', 'margin-left': '15%'}),
)

app.layout = html.Div(children=[
    dcc.Store(id='memory'),
    header,
    input_groups,
    submit_button,
    html.Br(),
    jumbotron,
    html.Br(),
    html.Br()
])

@app.callback([Output(component_id='first_jumbotron', component_property='children'),
                Output(component_id='second_jumbotron', component_property='children'),
                Output(component_id='third_jumbotron', component_property='children'),
               Output(component_id='fourth_jumbotron', component_property='children'),
               Output(component_id='fifth_jumbotron', component_property='children'),
               Output(component_id='sixth_jumbotron', component_property='children')
               ],
              [Input("memory", "data")], prevent_initial_call=True)
def update(emb):

    dfs = get_user_rec(emb)

    j1 = get_jumbotron(dfs.iloc[0].title, dfs.iloc[0].description, dfs.iloc[0].location)
    j2 = get_jumbotron(dfs.iloc[1].title, dfs.iloc[1].description, dfs.iloc[1].location)
    j3 = get_jumbotron(dfs.iloc[2].title, dfs.iloc[2].description, dfs.iloc[2].location)
    j4 = get_jumbotron(dfs.iloc[3].title, dfs.iloc[3].description, dfs.iloc[3].location)
    j5 = get_jumbotron(dfs.iloc[4].title, dfs.iloc[4].description, dfs.iloc[4].location)
    j6 = get_jumbotron(dfs.iloc[5].title, dfs.iloc[5].description, dfs.iloc[5].location)

    return [j1, j2, j3, j4, j5, j6]

@app.callback(
    [Output("memory", "data")],
    [Input("update_output", "n_clicks"), State("username_input", "value"), State("title_input", "value"), State("desc_input", "value")],
    prevent_initial_call=True
)
def get_user_emb(n, username, title, desc):

    data = {'title': title,
            'description': desc,
            'skills': ' '}

    r = requests.post(url, json=data)
    return [r.json()['emb']]


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)