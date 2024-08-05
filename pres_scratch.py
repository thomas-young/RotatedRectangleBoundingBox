import plotly.graph_objects as go
import numpy as np

# Define the range of x and z values
x = np.linspace(-10, 10, 400)
z = np.linspace(-10, 10, 400)

# Create a meshgrid for x and z
X, Z = np.meshgrid(x, z)

# Define the equation
Y = 0.3 * X + 0.4 * Z + 0

# Create the interactive 3D plot
fig = go.Figure(data=[go.Surface(z=Y, x=X, y=Z)])

# Add title and labels
fig.update_layout(title='Graph of Y = 0.3x + 0.4z', 
                  scene = dict(
                    xaxis_title='X',
                    yaxis_title='Z',
                    zaxis_title='Y'),
                  autosize=False,
                  width=800, height=800)

# Save the plot to an HTML file
fig.write_html("./interactive_plot.html")
