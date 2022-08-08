import plotly.express as px
import plotly.graph_objs as go


class UpdateGraph:
    def __init__(self, title, plots_data: dict, cur_plot: str):
        self.fig = go.Figure()
        self.title = {
            'x': 0.5,
            'y': 0.98,
            'text': f'<b>{title}</b>',
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(family="Arial", size=30, color='#ffffff')
        }

        self.fig.update_layout(
            margin=dict(l=10, r=10, b=10, t=55, pad=0),
            title=self.title,
            template="plotly_dark",
            uirevision=True)
        if plots_data and cur_plot in plots_data:
            self.plot_data = plots_data[cur_plot]
            if self.plot_data['type'] == '2D':
                self.update_fig2d()
            elif self.plot_data['type'] == '3D':
                self.update_fig3d()

    def update_fig2d(self):
        for scatter_name, scatter_data in self.plot_data['scatters'].items():
            if all(len(scatter_data[key]) > 0 for key in ['x', 'y']):
                scatter_fig = go.Scatter(
                    x=scatter_data['x'], y=scatter_data['y'],
                    name=scatter_name,
                    mode=scatter_data['mode'])

                if 'line_size' in scatter_data:
                    scatter_fig.line = dict(width=scatter_data['line_size'])
                if 'marker_size' in scatter_data:
                    scatter_fig.marker = dict(size=scatter_data['marker_size'])
                if 'type' in scatter_data:
                    if isinstance(scatter_data["type"], int):
                        scatter_fig.line = dict(color=px.colors.qualitative.Light24[scatter_data["type"]])
                    scatter_fig.text = f'type: {scatter_data["type"]}'
                if 'state' in scatter_data:
                    scatter_fig.text += f'\nstates: {scatter_data["state"]}'
                if 'fill' in scatter_data:
                    scatter_fig.fill = 'toself'
                    scatter_fig.opacity = 0.9
                self.fig.add_traces([scatter_fig])
        self.fig.update_xaxes(title="X Position", scaleanchor='y')
        self.fig.update_yaxes(title="Y Position", constrain='domain')
        self.fig.update_layout(
            title=self.title,
            scene=dict(
                aspectratio=dict(x=1, y=1)
            ))

    def update_fig3d(self):
        max_vals, min_vals = [], []
        for scatter_name, scatter_data in self.plot_data['scatters'].items():
            if 'z' not in scatter_data:
                scatter_data['z'] = [0] * len(scatter_data['x'])
            if all(len(scatter_data[key]) > 0 for key in ['x', 'y', 'z']):
                max_vals.append(max([scatter_data['x'], scatter_data['y'], scatter_data['z']]))
                min_vals.append(min([scatter_data['x'], scatter_data['y'], scatter_data['z']]))

                scatter_fig = go.Scatter3d(
                    x=scatter_data['x'], y=scatter_data['y'], z=scatter_data['z'],
                    name=scatter_name,
                    mode='lines+markers')
                if 'type' in scatter_data:
                    scatter_fig.text = f'type: {scatter_data["type"]}',
                    scatter_fig.line = dict(color=px.colors.qualitative.Light24[scatter_data["type"]])
                self.fig.add_traces([scatter_fig])
        max_val, min_val = max(max_vals), min(min_vals)
        self.fig.update_layout(
            title=self.title,
            autosize=True,
            scene=dict(
                xaxis=dict(range=[min_val, max_val]),
                yaxis=dict(range=[min_val, max_val]),
                zaxis=dict(range=[min_val, max_val]),
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                aspectmode='cube'
            ))
