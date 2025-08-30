import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Data from the provided JSON - filtering to show the three implementation phases plus baseline
data = {
    "phases": ["Baseline", "Quick Wins", "Advanced", "Expert"],
    "cumulative_improvement": [0, 25, 45, 65],
    "time_hours": [0, 1.4, 4.2, 8.5],
    "techniques": [0, 4, 7, 10]
}

df = pd.DataFrame(data)

# Create subplot with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add cumulative improvement trace
fig.add_trace(
    go.Scatter(
        x=df['phases'],
        y=df['cumulative_improvement'],
        mode='lines+markers',
        name='Cumul Improv',
        marker=dict(size=16, color='#1FB8CD'),
        line=dict(width=5, color='#1FB8CD'),
        hovertemplate='<b>%{x}</b><br>Improv: %{y}%<br>Time: %{customdata[0]}h<br>Tech: %{customdata[1]}<extra></extra>',
        customdata=list(zip(df['time_hours'], df['techniques']))
    ),
    secondary_y=False,
)

# Add time investment trace
fig.add_trace(
    go.Scatter(
        x=df['phases'],
        y=df['time_hours'],
        mode='lines+markers',
        name='Time Invest',
        marker=dict(size=14, color='#DB4545'),
        line=dict(width=4, color='#DB4545', dash='dash'),
        hovertemplate='<b>%{x}</b><br>Time: %{y}h<br>Improv: %{customdata}%<extra></extra>',
        customdata=df['cumulative_improvement']
    ),
    secondary_y=True,
)

# Update layout
fig.update_layout(
    title='ML Progress Roadmap',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Set y-axes titles
fig.update_yaxes(title_text="Cumul Improv %", secondary_y=False)
fig.update_yaxes(title_text="Time Hours", secondary_y=True)

fig.update_xaxes(title_text="Phase", tickangle=0)

# Add some padding to x-axis
fig.update_xaxes(categoryorder='array', categoryarray=['Baseline', 'Quick Wins', 'Advanced', 'Expert'])

fig.update_traces(cliponaxis=False)

fig.write_image('improvement_roadmap.png')