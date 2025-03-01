import requests
import zipfile
import io
import pandas as pd
import streamlit as st 
import plotly.express as px
st.set_page_config(page_title='NHL Shot Analysis',layout='wide',page_icon='🏒')
st.sidebar.title('Filters')


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
st.markdown("<h1 style='text-align: center; font-size: 100px;'>NHL Shot Analysis</h1>", unsafe_allow_html=True)
season = st.selectbox('Select a season', list(range(2007,2025)))
if season:
    url = f"https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Step 2: Open the ZIP file from the response content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # List the files inside the ZIP to find the CSV file
            print("Files in the ZIP archive:", z.namelist())
            
            # Step 3: Read the CSV file into a pandas DataFrame
            # Assuming the CSV file is the first in the list or you can specify the filename
            csv_filename = z.namelist()[0]  # Update this if you know the exact filename
            with z.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
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
        filterbyplayer = st.sidebar.toggle('Filter by Player')
        if filterbyplayer:
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

        # Display the plot in Streamlit
        if filter != 'Team':
            display_player_image(player_id=playerid,width2=300,caption2=f'{playername}')
        else:
            teamname = teamname.replace('.','')
            if teamname == 'SJ':
                teamname = 'SJS'
            elif teamname == 'TB':
                teamname = 'TBL'
            elif teamname == 'LA':
                teamname = 'LAK'
            display_player_image2(player_id=teamname,width2=400,caption2='')
        col1,col2 = st.columns(2)
        hoverlabel = df.apply(lambda row:f"""
                <b>Shooter:</b> {row['shooterName']}<br>
                <b>Goalie:</b> {row['goalieNameForShot']}<br>
                <b>Event:</b> {row['event']}<br>
                <b>Shot Distance:</b> {round(row['shotDistance'],2)} ft<br>
                <b>Shot Type:</b> {row['shotType']}<br>
                <b>Period:</b> {row['period']}<br>
                <b>Time:</b> {round(row['time']/60)}:{row['time']%60:02d}<br>
                <b>Game:</b> {row['homeTeamCode']} vs {row['awayTeamCode']}<br>
                <b>Season:</b> {row['season']}<br>

                """, axis=1)
        if rinktype == 'Half':
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=-df['xCordAdjusted'],
                    x=-df['yCordAdjusted'],
                    mode='markers',
                    showlegend=False,
                    marker=dict(color=df['color'], size=10,opacity=0.5),
                    name='End Points',
                    hovertext=hoverlabel,
                        hoverinfo='text',
                ))
                create_hockey_rink(fig,setting='dzone',vertical=True)
                st.plotly_chart(fig)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(
                    x=-df['yCordAdjusted'],  # Adjusted x coordinates
                    y=-df['xCordAdjusted'],  # Adjusted y coordinates
                    colorscale='hot',  # You can choose other colorscales
                    # ncontours=20,  # Number of contour levels
                    colorbar=dict(title="Density"),
                    showscale=False,
                    opacity=0.6,  # To blend with the scatter plot
                    name='Density',
                    hoverinfo='none'  # Don't show hover for the density contour
                    # ,zmin=0,
                    # zmax=1
                ))
                create_hockey_rink(fig,setting='dzone',vertical=True)
                st.plotly_chart(fig)
        else:
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(
                    y=df['yCord'],  # Adjusted x coordinates
                    x=df['xCord'],  # Adjusted y coordinates
                    colorscale='hot',  # You can choose other colorscales
                    # ncontours=20,  # Number of contour levels
                    colorbar=dict(title="Density"),
                    showscale=False,
                    opacity=0.6,  # To blend with the scatter plot
                    name='Density',
                    hoverinfo='none'  # Don't show hover for the density contour
                ))
                create_hockey_rink(fig,setting='full',vertical=False)
                st.plotly_chart(fig)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['xCord'],
                    y=df['yCord'],
                    mode='markers',
                    showlegend=False,
                    marker=dict(color=df['color'], size=10,opacity=0.5),
                    name='End Points',
                    hovertext=hoverlabel,
                        hoverinfo='text',
                ))
                create_hockey_rink(fig,setting='full',vertical=False)
                st.plotly_chart(fig)
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
            shootTime = df.groupby(['shotType','shooterTimeOnIce'])['shotID'].count().reset_index().rename(columns={'shotID':'Shots'})
            fig = px.line(shootTime,x='shooterTimeOnIce',y='Shots',color='shotType',title='Number of Shots Over Time')
            st.plotly_chart(fig)
        with c6:
            shotsPerGame = df.groupby(['game_id','period'])['shotID'].count().reset_index().rename(columns={'shotID':'Shots'})
            fig = px.line(shotsPerGame,x='game_id',y='Shots',color='period',title='Shots Per Game')
            st.plotly_chart(fig)
st.sidebar.markdown(f'Data from [moneypuck.com](https://moneypuck.com/)')

