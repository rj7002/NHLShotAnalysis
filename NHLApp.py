import requests
import zipfile
import io
import pandas as pd
import streamlit as st 
import plotly.express as px
import json
import re
st.set_page_config(page_title='NHL Shot Analysis',layout='wide',page_icon='🏒')
st.sidebar.title('Filters')

API_URL = "https://api.sambanova.ai/v1/chat/completions"
API_KEY = st.secrets["api"]["key"]
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def get_team_scouting_report(player_summary):
        # Construct payload
        payload = {
            "model": "DeepSeek-R1-Distill-Llama-70B",
            "messages": [
                # {"role": "system", "content": "You are an expert hockey scout analyzing data from a shot attempt from an NHL game. Give insights about the shot based on the data given and analyze the shot in detail."},
                {
  "role": "system",
  "content": "You are a professional NHL hockey scout analyzing a single shot attempt from a game. Based on the provided shot data, generate a standardized scouting report with the following structure:\n\n1. **Shot Overview** – Summarize key shot details (e.g., shooter, shot type, distance, angle, location).\n2. **Scoring Chance Assessment** – Rate the shot’s danger level (low / medium / high) and explain your reasoning.\n3. **Goaltender Analysis** – Evaluate the goalie’s likely positioning and difficulty in stopping the shot.\n4. **Play Context** – Describe how the shot likely developed (e.g., off a rush, rebound, cycle).\n5. **Scout’s Evaluation** – Give a concise expert opinion on the shot quality, player decision-making, and any notable observations.\n\nAlways respond using this five-part format. Keep tone professional, analytical, and concise."
},

                {"role": "user", "content": player_summary}
            ],
            "max_tokens": 10000, 
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "stream": True  
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)

        if response.status_code == 200:
            output_placeholder = st.empty()
            full_report = ""

            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode('utf-8')
                        if data.startswith("data:"):  
                            response_data = data.split("data:")[1].strip()
                            
                            if not response_data:
                                continue
                            
                            chunk = json.loads(response_data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            
                            full_report += content

                            full_report = re.sub(r"<think>.*?</think>", "", full_report, flags=re.DOTALL).strip()

                            output_placeholder.code(full_report, language="markdown")

                    except Exception as e:
                        st.error(f"Error processing stream: {e}")
                        break
            return full_report
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
        
# Step 1: Download the ZIP file from the URL
def display_player_image(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://assets.nhle.com/mugs/nhl/latest/{player_id}.png"
    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{caption2}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://assets.nhle.com/mugs/nhl/latest/8481773.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
def display_player_image2(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://assets.nhle.com/logos/nhl/svg/{player_id}_dark.svg"

    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{caption2}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://assets.nhle.com/mugs/nhl/latest/8481773.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

finalfeats = ['season',  'time',
       'timeSinceLastEvent', 'shotGoalieFroze', 'shotPlayStopped',
       'awayTeamGoals', 'xCord', 'yCord', 'xCordAdjusted', 'yCordAdjusted',
       'shotAngle', 'shotAngleAdjusted', 'shotAnglePlusRebound',
       'shotAngleReboundRoyalRoad', 'shotDistance', 'shotOnEmptyNet',
       'shotRebound', 'shotAnglePlusReboundSpeed', 'speedFromLastEvent',
       'lastEventxCord', 'lastEventyCord', 'distanceFromLastEvent',
       'lastEventShotAngle', 'lastEventShotDistance', 'homeEmptyNet',
       'awayEmptyNet', 'homeSkatersOnIce', 'awaySkatersOnIce',
       'awayPenalty1TimeLeft', 'awayPenalty1Length', 'homePenalty1TimeLeft',
       'homePenalty1Length', 'playerNumThatDidEvent',
       'lastEventxCord_adjusted', 'lastEventyCord_adjusted',
       'timeSinceFaceoff', 'shooterPlayerId', 'shooterTimeOnIce',
       'shooterTimeOnIceSinceFaceoff', 'shootingTeamForwardsOnIce',
       'shootingTeamDefencemenOnIce', 'shootingTeamAverageTimeOnIce',
       'shootingTeamAverageTimeOnIceOfForwards',
       'shootingTeamAverageTimeOnIceOfDefencemen', 'shootingTeamMaxTimeOnIce',
       'shootingTeamMaxTimeOnIceOfForwards',
       'shootingTeamMaxTimeOnIceOfDefencemen', 'shootingTeamMinTimeOnIce',
       'shootingTeamMinTimeOnIceOfForwards',
       'shootingTeamMinTimeOnIceOfDefencemen',
       'shootingTeamAverageTimeOnIceSinceFaceoff',
       'shootingTeamAverageTimeOnIceOfForwardsSinceFaceoff',
       'shootingTeamAverageTimeOnIceOfDefencemenSinceFaceoff',
       'shootingTeamMaxTimeOnIceSinceFaceoff',
       'shootingTeamMaxTimeOnIceOfForwardsSinceFaceoff',
       'shootingTeamMaxTimeOnIceOfDefencemenSinceFaceoff',
       'shootingTeamMinTimeOnIceSinceFaceoff',
       'shootingTeamMinTimeOnIceOfForwardsSinceFaceoff',
       'shootingTeamMinTimeOnIceOfDefencemenSinceFaceoff',
       'defendingTeamForwardsOnIce', 'defendingTeamDefencemenOnIce',
       'defendingTeamAverageTimeOnIce',
       'defendingTeamAverageTimeOnIceOfForwards',
       'defendingTeamAverageTimeOnIceOfDefencemen',
       'defendingTeamMaxTimeOnIce', 'defendingTeamMaxTimeOnIceOfForwards',
       'defendingTeamMaxTimeOnIceOfDefencemen', 'defendingTeamMinTimeOnIce',
       'defendingTeamMinTimeOnIceOfForwards',
       'defendingTeamMinTimeOnIceOfDefencemen',
       'defendingTeamAverageTimeOnIceSinceFaceoff',
       'defendingTeamAverageTimeOnIceOfForwardsSinceFaceoff',
       'defendingTeamAverageTimeOnIceOfDefencemenSinceFaceoff',
       'defendingTeamMaxTimeOnIceSinceFaceoff',
       'defendingTeamMaxTimeOnIceOfForwardsSinceFaceoff',
       'defendingTeamMaxTimeOnIceOfDefencemenSinceFaceoff',
       'defendingTeamMinTimeOnIceSinceFaceoff',
       'defendingTeamMinTimeOnIceOfForwardsSinceFaceoff',
       'defendingTeamMinTimeOnIceOfDefencemenSinceFaceoff', 'offWing',
       'arenaAdjustedShotDistance', 'arenaAdjustedXCord', 'arenaAdjustedYCord',
       'arenaAdjustedYCordAbs', 'timeDifferenceSinceChange',
       'averageRestDifference', 'isHomeTeam', 'shotWasOnGoal',
       'arenaAdjustedXCordABS']
@st.cache_data
def load_data(season):
    url = f"https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip"
    response = requests.get(url)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            
            csv_filename = z.namelist()[0] 
            with z.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                return df
st.markdown("<h1 style='text-align: center; font-size: 100px;'>NHL Shot Analysis</h1>", unsafe_allow_html=True)
season = st.selectbox('Select a season', list(range(2007,2026)))
if season:
        df = load_data(season)
        df['opposingTeam'] = df.apply(lambda row: row['awayTeamCode'] if row['teamCode'] != row['awayTeamCode'] else row['homeTeamCode'], axis=1)
        filter = st.selectbox('Filter by',['Shooter','Goalie','Team'])
        if filter == 'Shooter':
            playernames = df['shooterName'].unique()
            playername = st.selectbox('Select a player',playernames)
            df = df[df['shooterName'] == playername]
        elif filter == 'Goalie':
            playernames = df['goalieNameForShot'].unique()
            playername = st.selectbox('Select a player',playernames)
            df = df[df['goalieNameForShot'] == playername]
        else:
            teamnames = df['teamCode'].unique()
            teamname = st.selectbox('Select a player',teamnames)
            df = df[df['teamCode'] == teamname]
        uniqueshots = df['shotType'].unique()
        shotType = st.sidebar.multiselect('Select shot type',uniqueshots,default=uniqueshots.tolist())
        df = df[df['shotType'].isin(shotType)]
        time_min, time_max = st.sidebar.slider(
            "Select Time Range", 
            min_value=int(df['time'].min()), 
            max_value=int(df['time'].max()), 
            value=(int(df['time'].min()), int(df['time'].max())), 
            step=1
        )
        df = df[(df['time'] >= time_min) & (df['time'] <= time_max)]

        # Shot Distance filter slider
        shot_distance_min, shot_distance_max = st.sidebar.slider(
            "Select Shot Distance Range", 
            min_value=int(df['shotDistance'].min()), 
            max_value=int(df['shotDistance'].max()), 
            value=(int(df['shotDistance'].min()), int(df['shotDistance'].max())), 
            step=1
        )
        df = df[(df['shotDistance'] >= shot_distance_min) & (df['shotDistance'] <= shot_distance_max)]

        shot_angle_min, shot_angle_max = st.sidebar.slider(
            "Select Shot Angle Range", 
            min_value=int(df['shotAngle'].min()), 
            max_value=int(df['shotAngle'].max()), 
            value=(int(df['shotAngle'].min()), int(df['shotAngle'].max())), 
            step=1
        )
        df = df[(df['shotAngle'] >= shot_angle_min) & (df['shotAngle'] <= shot_angle_max)]

        # Opponent Team filter (multiselect)
        team_options = df['awayTeamCode'].unique()
        selected_teams = st.sidebar.multiselect("Select Opponent Team(s)", team_options, default=team_options.tolist())
        df = df[df['awayTeamCode'].isin(selected_teams)]

        # Event filter (selectbox)
        event_options = df['event'].unique()
        selected_event = st.sidebar.multiselect("Select Event Type", event_options,default=event_options.tolist())
        df = df[df['event'].isin(selected_event)]

        period_options = df['period'].unique()
        selected_period = st.sidebar.multiselect("Select Period", period_options,default=period_options.tolist())
        df = df[df['period'].isin(selected_period)]
        xGoal = st.sidebar.checkbox('xG')
        if st.sidebar.toggle('Filter by players'):
            if filter == 'Shooter':
                player_options = df['goalieNameForShot'].unique()
                selected_player = st.sidebar.multiselect("Select Goalie", player_options,default=player_options.tolist())
                df = df[df['goalieNameForShot'].isin(selected_player)]
            elif filter == 'Goalie':
                player_options = df['shooterName'].unique()
                selected_player = st.sidebar.multiselect("Select Shooter", player_options,default=player_options.tolist())
                df = df[df['shooterName'].isin(selected_player)]

        if filter == 'Shooter':
            playerid = df['shooterPlayerId'].iloc[0]
        elif filter == 'Goalie':
            playerid = df['goalieIdForShot'].iloc[0]
        rinktype = st.selectbox('Rink Type',['Full','Half'])
        import numpy as np

    
        import plotly.graph_objects as go
        import pandas as pd
        import streamlit as st

    

        import numpy as np
        import plotly.graph_objects as go

        def create_hockey_rink(fig,setting, vertical):
            '''
            Function to plot rink in Plotly. Takes 2 arguments :

            setting : full (default) for full ice, offense positive half of the ice, ozone positive quarter of ice, defense for negative half of the ice, dzone for negative quarter of the ice, and neutral for the neutral zone
            vertical : True if you want a vertical rink, False (default) is for an horizontal rink

            '''

            def faceoff_circle(x, y):
                theta = np.linspace(0, 2*np.pi, 300)
                # Outer circle
                x_outer = x + 15*np.cos(theta)
                y_outer = y + 15*np.sin(theta)
                outer_circle = go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(width=2, color='red'), showlegend=False, hoverinfo='skip')
                # Inner circle
                x_inner = x + np.cos(theta)
                y_inner = y + np.sin(theta)
                inner_circle = go.Scatter(x=x_inner, y=y_inner, mode='lines', fill='toself', fillcolor='rgba(255, 0, 0, 0.43)', line=dict(color='rgba(255, 0, 0, 1)', width=2), showlegend=False, hoverinfo='skip')
                
                
                return [outer_circle, inner_circle]  #segments
            

            if vertical :
                setting_dict = {
                    "full" : [-101, 101],
                    "offense" : [0, 101],
                    "ozone" : [25, 101],
                    "defense" : [-101, 0],
                    "dzone" : [-101, -25],
                    "neutral" : [-25,25]
                }
                

                fig.update_layout(xaxis=dict(range=[-42.6, 42.6], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"), yaxis=dict(range=setting_dict[setting], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
                                showlegend=False, autosize=True, template="plotly_white")
                fig.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1,
                )
                def goal_crease(flip=1):
                    x_seq = np.linspace(-4, 4, 100)
                    x_goal = np.concatenate(([-4], x_seq, [4]))
                    y_goal = flip * np.concatenate(([89], 83 + x_seq**2/4**2*1.5, [89]))
                    goal_crease = go.Scatter(x=x_goal, y=y_goal, fill='toself', fillcolor='rgba(173, 216, 230, 0.3)', line=dict(color='red'))
                    return goal_crease

                # Outer circle
                theta = np.linspace(0, 2*np.pi, 300)
                x_outer = 15 * np.cos(theta)
                y_outer = 15 * np.sin(theta)
                fig.add_trace(go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(color='royalblue', width=2), showlegend=False, hoverinfo='skip'))
                # Inner circle
                theta2 = np.linspace(np.pi/2, 3*np.pi/2, 300)
                x_inner = 42.5 + 10 * np.cos(theta2)
                y_inner = 10 * np.sin(theta2)
                fig.add_trace(go.Scatter(x=x_inner, y=y_inner, mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='skip'))
                # Rink boundaries
                fig.add_shape(type='rect', xref='x', yref='y', x0=-42.5, y0=25, x1=42.5, y1=26, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
                fig.add_shape(type='rect', xref='x', yref='y', x0=-42.5, y0=-25, x1=42.5, y1=-26, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
                fig.add_shape(type='rect', xref='x', yref='y', x0=-42.5, y0=-0.5, x1=42.5, y1=0.5, line=dict(color='red', width=2), fillcolor='red')
                
                # Goal crease
                fig.add_trace(goal_crease())
                fig.add_trace(goal_crease(-1))
                # Goal lines
                goal_line_extreme = 42.5 - 28 + np.sqrt(28**2 - (28-11)**2)
                fig.add_shape(type='line', xref='x', yref='y', x0=-goal_line_extreme, y0=89, x1=goal_line_extreme, y1=89, line=dict(color='red', width=2))
                fig.add_shape(type='line', xref='x', yref='y', x0=-goal_line_extreme, y0=-89, x1=goal_line_extreme, y1=-89, line=dict(color='red', width=2))

                # Faceoff circles
                fig.add_traces(faceoff_circle(-22, 69))
                fig.add_traces(faceoff_circle(22, 69))
                fig.add_traces(faceoff_circle(-22, -69))
                fig.add_traces(faceoff_circle(22, -69))
                # Sidelines
                theta_lines = np.linspace(0, np.pi/2, 20)
                x_lines1 = np.concatenate(([-42.5], -42.5 + 28 - 28*np.cos(theta_lines), 42.5 - 28 + 28*np.cos(np.flip(theta_lines))))
                y_lines1 = np.concatenate(([15], 72 + 28*np.sin(theta_lines), 72 + 28*np.sin(np.flip(theta_lines))))
                x_lines2 = np.concatenate(([-42.5], -42.5 + 28 - 28*np.cos(theta_lines), 42.5 - 28 + 28*np.cos(np.flip(theta_lines))))
                y_lines2 = np.concatenate(([15], -72 - 28*np.sin(theta_lines), -72 - 28*np.sin(np.flip(theta_lines))))
                fig.add_trace(go.Scatter(x=x_lines1, y=y_lines1, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=x_lines2, y=y_lines2, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_shape(type='line', xref='x', yref='y', x0=42.5, y0=-72.5, x1=42.5, y1=72.5, line=dict(color='black', width=2))
                fig.add_shape(type='line', xref='x', yref='y', x0=-42.5, y0=-72.5, x1=-42.5, y1=72.5, line=dict(color='black', width=2))
                
            else : 
                setting_dict = {
                    "full" : [-101, 101],
                    "offense" : [0, 101],
                    "ozone" : [25, 101],
                    "defense" : [-101, 0],
                    "dzone" : [-101, -25]
                }
                fig.update_layout(xaxis=dict(range=setting_dict[setting], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(range=[-42.6, 42.6], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
                                showlegend=True, autosize =True, template="plotly_white")
                fig.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1,
                )
                def goal_crease(flip=1):
                    y_seq = np.linspace(-4, 4, 100)
                    y_goal = np.concatenate(([-4], y_seq, [4]))
                    x_goal = flip * np.concatenate(([89], 83 + y_seq**2/4**2*1.5, [89]))
                    goal_crease = go.Scatter(x=x_goal, y=y_goal, fill='toself', fillcolor='rgba(173, 216, 230, 0.3)', line=dict(color='red'), showlegend=False, hoverinfo='skip')
                    return goal_crease
                
                # Outer circle
                theta = np.linspace(0, 2 * np.pi, 300)
                x_outer = 15 * np.sin(theta)
                y_outer = 15 * np.cos(theta)
                fig.add_trace(go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(color='royalblue', width=2), showlegend=False, hoverinfo='skip'))
                # Inner circle
                theta2 = np.linspace(3 * np.pi / 2, np.pi / 2, 300)  # Update theta2 to rotate the plot by 180 degrees
                x_inner = 10 * np.sin(theta2)  # Update x_inner to rotate the plot by 180 degrees
                y_inner = -42.5 - 10 * np.cos(theta2)  # Update y_inner to rotate the plot by 180 degrees
                fig.add_trace(go.Scatter(x=x_inner, y=y_inner, mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='skip'))
                
                # Rink boundaries
                fig.add_shape(type='rect', xref='x', yref='y', x0=25, y0=-42.5, x1=26, y1=42.5, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
                fig.add_shape(type='rect', xref='x', yref='y', x0=-25, y0=-42.5, x1=-26, y1=42.5, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
                fig.add_shape(type='rect', xref='x', yref='y', x0=-0.5, y0=-42.5, x1=0.5, y1=42.5, line=dict(color='red', width=2), fillcolor='red')
                # Goal crease
                fig.add_trace(goal_crease())
                fig.add_trace(goal_crease(-1))
                # Goal lines
                goal_line_extreme = 42.5 - 28 + np.sqrt(28 ** 2 - (28 - 11) ** 2)
                fig.add_shape(type='line', xref='x', yref='y', x0=89, y0=-goal_line_extreme, x1=89, y1=goal_line_extreme, line=dict(color='red', width=2))
                fig.add_shape(type='line', xref='x', yref='y', x0=-89, y0=-goal_line_extreme, x1=-89, y1=goal_line_extreme, line=dict(color='red', width=2))
                # Faceoff circles
                fig.add_traces(faceoff_circle(-69, -22))
                fig.add_traces(faceoff_circle(-69, 22))
                fig.add_traces(faceoff_circle(69, -22))
                fig.add_traces(faceoff_circle(69, 22))
                # Sidelines
                theta_lines = np.linspace(0, np.pi / 2, 20)
                x_lines1 = np.concatenate(([15], 72 + 28 * np.sin(theta_lines), 72 + 28 * np.sin(np.flip(theta_lines))))
                y_lines1 = np.concatenate(([-42.5], -42.5 + 28 - 28 * np.cos(theta_lines), 42.5 - 28 + 28 * np.cos(np.flip(theta_lines))))
                x_lines2 = np.concatenate(([15], -72 - 28 * np.sin(theta_lines), -72 - 28 * np.sin(np.flip(theta_lines))))
                y_lines2 = np.concatenate(([-42.5], -42.5 + 28 - 28 * np.cos(theta_lines), 42.5 - 28 + 28 * np.cos(np.flip(theta_lines))))
                fig.add_trace(go.Scatter(x=x_lines1, y=y_lines1, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=x_lines2, y=y_lines2, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_shape(type='line', xref='x', yref='y', x0=-72.5, y0=-42.5, x1=72.5, y1=-42.5, line=dict(color='black', width=2))
                fig.add_shape(type='line', xref='x', yref='y', x0=-72.5, y0=42.5, x1=72.5, y1=42.5, line=dict(color='black', width=2))
            return fig
        # Create the rink
        # fig = create_hockey_rink()
        # df = df[df['event'] == 'SHOT']
        df['color'] = np.where(df['event'] == 'GOAL','gold',np.where(df['event'] == 'SHOT', 'green', 'red'))
        modelinput = df[finalfeats]
        import pickle
        with open('nhlshotmodel.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        preds = loaded_model.predict_proba(modelinput)[:, 1]
        df['xG'] = preds
        if xGoal:
            o = df['xG']
        else:
            o = 0.5
        # Display the plot in Streamlit
        if filter != 'Team':
            byPlayer = df.groupby('shooterName').agg({'xG':'sum','goal':'sum','event':'count'}).reset_index()
            byPlayer['xGOE'] = byPlayer['goal'] - byPlayer['xG']
            byPlayer['xG%'] = byPlayer['xG']/byPlayer['event']
            byPlayer['G%'] = byPlayer['goal']/byPlayer['event']
            byPlayer = byPlayer.rename(columns={'event':'Shots','goal':'Goals'})
            byPlayer = byPlayer.sort_values('xGOE',ascending=False)
            display_player_image(player_id=playerid,width2=300,caption2=f'{playername}')
        else:
            byPlayer = df.groupby('teamCode').agg({'xG':'sum','goal':'sum','event':'count'}).reset_index()
            byPlayer['xGOE'] = byPlayer['goal'] - byPlayer['xG']
            byPlayer['xG%'] = byPlayer['xG']/byPlayer['event']
            byPlayer['G%'] = byPlayer['goal']/byPlayer['event']
            byPlayer = byPlayer.rename(columns={'event':'Shots','goal':'Goals'})
            byPlayer = byPlayer.sort_values('xGOE',ascending=False)
            teamname = teamname.replace('.','')
            if teamname == 'SJ':
                teamname = 'SJS'
            elif teamname == 'TB':
                teamname = 'TBL'
            elif teamname == 'LA':
                teamname = 'LAK'
            display_player_image2(player_id=teamname,width2=400,caption2='')
        col1,col2 = st.columns(2)
        with col1:
            st.subheader(f'Shots: {byPlayer["Shots"].iloc[0]}')
            st.subheader(f'Goals: {byPlayer["Goals"].iloc[0]}')
            st.subheader(f'xGoals: {round(byPlayer["xG"].iloc[0],2)}')
        with col2:
            st.subheader(f'Goal%: {round(byPlayer["G%"].iloc[0]*100,2)}%')
            st.subheader(f'xGoal%: {round(byPlayer["xG%"].iloc[0]*100,2)}%')
            st.subheader(f'xGOE: {round(byPlayer["xGOE"].iloc[0],2)}')
        hoverlabel = df.apply(lambda row:f"""
                <b>Shooter:</b> {row['shooterName']}<br>
                <b>Goalie:</b> {row['goalieNameForShot']}<br>
                <b>Event:</b> {row['event']}<br>
                <b>Zone:</b> {row['location']}<br>
                <b>Angle:</b> {row['shotAngle']}<br>
                <b>Shot Distance:</b> {round(row['shotDistance'],2)} ft<br>
                <b>Shot Type:</b> {row['shotType']}<br>
                <b>Period:</b> {row['period']}<br>
                <b>Time:</b> {round(row['time']/60)}:{row['time']%60:02d}<br>
                <b>Game:</b> {row['homeTeamCode']} vs {row['awayTeamCode']}<br>
                <b>Season:</b> {row['season']}<br>


                """, axis=1)
        df['symbol'] = np.where(df['goal'] == 1, 'star', np.where(df['event'] == 'MISS', 'x', 'circle'))
        if rinktype == 'Half':
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=-df['xCordAdjusted'],
                    x=-df['yCordAdjusted'],
                    mode='markers',
                    marker=dict(color=df['color'], size=10,opacity=o,symbol=df['symbol']),
                    showlegend=False,
                    name='End Points',
                    hovertext=hoverlabel,
                        hoverinfo='text',
                ))
                create_hockey_rink(fig,setting='dzone',vertical=True)
                data = st.plotly_chart(fig,on_select='rerun')
                x = data["selection"]["points"][0]["x"]
                y = data["selection"]["points"][0]["y"]

                modeldata = df[(df['xCord'] == x) & (df['yCord'] == y)]
                row_values_with_columns = [f"{col}: {value}" for col, value in modeldata.items()]
                data_summary = "\n".join(row_values_with_columns)


                # xG = modeldata['xGoal'].iloc[0]
                # modelinput = modeldata[finalfeats]
                # import pickle
                # with open('/Users/ryan/Desktop/ShotQuality/nhlshotmodel.pkl', 'rb') as f:
                #     loaded_model = pickle.load(f)
                # pred = loaded_model.predict_proba(modelinput)[:, 1][0]
                # if pred*100 < 0:
                #     predstr = round(100*pred,8)
                # else:
                #     predstr = round(100*pred,2)
                # st.write(f"xGAct {xG*100}")
                st.subheader(f"xG: {modeldata['xG'].iloc[0]*100}%")
                report = get_team_scouting_report(data_summary)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(
                    x=-df['yCordAdjusted'],  # Adjusted x coordinates
                    y=-df['xCordAdjusted'],  # Adjusted y coordinates
                    # colorscale='Magma',  # You can choose other colorscales
                    # ncontours=20,  # Number of contour levels
                    colorbar=dict(title="Density"),
                    showscale=False,
                    # opacity=0.6,  # To blend with the scatter plot
                    name='Density',
                    hoverinfo='none'  # Don't show hover for the density contour
                    ,
                    colorscale = [
                        [0, 'rgb(255, 255, 255)'],  # White
                         [0.01, 'rgb(0, 0, 0)'],  # Black
                        [0.25, 'rgb(169, 169, 169)'],  # Gray
                        [0.5, 'rgb(255, 0, 0)'],    # Red
                        [1, 'rgb(255, 255, 0)']     # Yellow
                    ]


                ))
                create_hockey_rink(fig,setting='dzone',vertical=True)
                st.plotly_chart(fig)
        else:
            with col2:
                fig = go.Figure()
                df1 = df[df['xCord'] < 0]
                df2 = df[df['xCord'] >= 0]
                fig.add_trace(go.Histogram2dContour(
                    y=df1['yCord'],  # Adjusted x coordinates
                    x=df1['xCord'],  # Adjusted y coordinates
                    # colorscale='Magma',  # You can choose other colorscales
                    # ncontours=20,  # Number of contour levels
                    colorbar=dict(title="Density"),
                    # opacity=0.6,  # To blend with the scatter plot
                    name='Density',
                    showscale=False,
                    hoverinfo='none'  # Don't show hover for the density contour
                    ,
                    colorscale = [
                        [0, 'rgb(255, 255, 255)'],  # White
                          [0.01, 'rgb(0, 0, 0)'],  # Black
                        [0.25, 'rgb(169, 169, 169)'],  # Gray
                        [0.5, 'rgb(255, 0, 0)'],    # Red
                        [1, 'rgb(255, 255, 0)']     # Yellow
                    ]


                ))
                fig.add_trace(go.Histogram2dContour(
                    y=df2['yCord'],  # Adjusted x coordinates
                    x=df2['xCord'],  # Adjusted y coordinates
                    # colorscale='Magma',  # You can choose other colorscales
                    # ncontours=20,  # Number of contour levels
                    colorbar=dict(title="Density"),
                    # opacity=0.6,  # To blend with the scatter plot
                    name='Density',
                    showscale=False,
                    hoverinfo='none'  # Don't show hover for the density contour
                    ,
                    colorscale = [
                        [0, 'rgb(255, 255, 255)'],  # White
                          [0.01, 'rgb(0, 0, 0)'],  # Black
                        [0.25, 'rgb(169, 169, 169)'],  # Gray
                        [0.5, 'rgb(255, 0, 0)'],    # Red
                        [1, 'rgb(255, 255, 0)']     # Yellow
                    ]


                ))
                create_hockey_rink(fig,setting='full',vertical=False)
                st.plotly_chart(fig)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['xCord'],
                    y=df['yCord'],
                    mode='markers',
                    marker=dict(color=df['color'], size=10,opacity=o,symbol=df['symbol']),
                    name='End Points',
                    hovertext=hoverlabel,
                    showlegend=False,
                        hoverinfo='text',
                ))
                create_hockey_rink(fig,setting='full',vertical=False)
                data = st.plotly_chart(fig,on_select='rerun')
                if data and 'selection' in data and 'points' in data['selection'] and len(data['selection']['points']) > 0:
                    x = data["selection"]["points"][0]["x"]
                    y = data["selection"]["points"][0]["y"]

                    modeldata = df[(df['xCord'] == x) & (df['yCord'] == y)]
                    row_values_with_columns = [f"{col}: {value}" for col, value in modeldata.items()]
                    data_summary = "\n".join(row_values_with_columns)
                    # xG = modeldata['xGoal'].iloc[0]
                    # modelinput = modeldata[finalfeats]
                    # import pickle
                    # with open('/Users/ryan/Desktop/ShotQuality/nhlshotmodel.pkl', 'rb') as f:
                    #     loaded_model = pickle.load(f)
                    # pred = loaded_model.predict_proba(modelinput)[:, 1][0]
                    # if pred*100 < 0:
                    #     predstr = round(100*pred,8)
                    # else:
                    #     predstr = round(100*pred,2)
                    # st.write(f"xGAct {xG*100}")
                    st.subheader(f"xG: {modeldata['xG'].iloc[0]*100}%")
                    report = get_team_scouting_report(data_summary)
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df,x='shotDistance',title='Histogram of Shot Distance',color='shotType',opacity=0.5,color_discrete_map={0: 'red', 1: 'green'})
            fig.update_traces(marker=dict(line=dict(color="black", width=2)))
            st.plotly_chart(fig)
        with c2:
            if filter == 'Shooter':
                goalie_shots = df['goalieNameForShot'].value_counts().reset_index().head(25)
                goalie_shots.columns = ['Goalie', 'Shot Count']

                # Create a pie chart
                fig = px.pie(goalie_shots, 
                            values='Shot Count', 
                            names='Goalie', 
                            title="Top 25 Goalies Shot On")
            else:
                goalie_shots = df['shooterName'].value_counts().reset_index().head(25)
                goalie_shots.columns = ['Shooter', 'Shot Count']

                # Create a pie chart
                fig = px.pie(goalie_shots, 
                            values='Shot Count', 
                            names='Shooter', 
                            title="Top 25 Shooters")
            st.plotly_chart(fig)
        c3,c4 = st.columns(2)
        with c3:
            fig = px.histogram(df,x='shotAngle',title='Histogram of Shot Angle',color='shotType',opacity=0.5,color_discrete_map={0: 'red', 1: 'green'})
            fig.update_traces(marker=dict(line=dict(color="black", width=2)))
            st.plotly_chart(fig)
        with c4:
            eventgroup = df.groupby(['shotType','event'])['shotID'].agg('count').reset_index().rename(columns={'shotID':'Count'})
            fig = px.bar(eventgroup,x='shotType',y='Count',color='event',title='Shot Type Count')
            fig.update_traces(marker=dict(line=dict(color="black", width=2)))
            st.plotly_chart(fig)
        c5,c6 = st.columns(2)
        with c5:
            shootTime = df.groupby(['shotType']).agg({'goal':'mean','xG':'mean'}).reset_index().rename(columns={'shotType':'Shot Type'})
            fig = px.bar(shootTime,x='Shot Type',y='xG',title='Average xG by Shot Type',color='Shot Type')
            fig.add_trace(go.Bar(x=shootTime['Shot Type'], y=shootTime['goal'], name='Actual Goals', marker_color='red'))
            fig.update_traces(marker=dict(line=dict(color="black", width=2)))
            st.plotly_chart(fig)
        with c6:
            xFGvsG = df.groupby(['opposingTeam','game_id']).agg({'goal':'sum','xG':'sum'}).reset_index().rename(columns={'opposingTeam':'Opponent'})
            fig = px.line(xFGvsG,x='Opponent',y='xG',title='Expected vs Actual Goals Per Opponent')
            fig.add_trace(go.Scatter(x=xFGvsG['Opponent'], y=xFGvsG['goal'], mode='lines', name='Actual Goals', line=dict(color='red')))
            st.plotly_chart(fig)
st.sidebar.markdown(f'Data from [moneypuck.com](https://moneypuck.com/)')
