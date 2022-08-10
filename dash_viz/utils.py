import plotly.express as px
import plotly.graph_objs as go


class CustomFigure:
    def __init__(self, title, plots_data: dict):
        self.figure = go.Figure()
        self.__init_base_layout(title)

        self.title = title
        self.plots_data = plots_data
        self.cur_plot_type = None
        self.cur_plot = None

    def __init_base_layout(self, title):
        self.figure.layout = {}
        self.figure.update_layout(
            margin=dict(l=10, r=10, b=10, t=55, pad=0),
            title={'x': 0.5, 'y': 0.98,
                   'text': f'<b>{title}</b>',
                   'xanchor': 'center', 'yanchor': 'top',
                   'font': dict(family="Arial", size=30, color='#ffffff')},
            template="plotly_dark",
            uirevision=True)

    def update(self, cur_plot):
        if self.plots_data and cur_plot in self.plots_data:
            self.cur_plot = cur_plot
            if self.cur_plot_type != self.plots_data[self.cur_plot]['type']:
                self.cur_plot_type = self.plots_data[self.cur_plot]['type']
                self.__update_layout()
            self.__update_data()
        return self.figure

    def __update_data(self):
        self.figure.data = []
        for scatter_name, scatter_data in self.plots_data[self.cur_plot]['scatters'].items():
            if all(key in scatter_data for key in ['x', 'y']):
                if self.plots_data[self.cur_plot]['type'] == '3D' and 'z' not in scatter_data:
                    scatter_data['z'] = [0] * len(scatter_data['x'])
                scatter_fig = self.__create_scatter(scatter_data, scatter_name)
                self.figure.add_traces([scatter_fig])

    def __update_layout(self):
        self.__init_base_layout(self.title)
        if self.cur_plot_type == '2D':
            self.figure.update_xaxes(title="X Position", scaleanchor='y')
            self.figure.update_yaxes(title="Y Position", constrain='domain')
            self.figure.update_layout(
                scene=dict(
                    aspectratio=dict(x=1, y=1)
                ))
        elif self.cur_plot_type == '3D':
            self.figure.update_layout(
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Z Position',
                    aspectratio=dict(x=1, y=1, z=1)
                ))

    def __create_scatter(self, scatter_data, scatter_name: str):
        if self.cur_plot_type == '3D':
            scatter_fig = go.Scatter3d(
                x=scatter_data['x'], y=scatter_data['y'], z=scatter_data['z'])
        else:
            scatter_fig = go.Scatter(
                x=scatter_data['x'], y=scatter_data['y'])

        scatter_fig.name = scatter_name
        scatter_fig.mode = scatter_data['mode']
        desc = ''
        if 'line_size' in scatter_data:
            scatter_fig.line.width = scatter_data['line_size']
        if 'line_type' in scatter_data:
            scatter_fig.line.dash = scatter_data['line_type']
        if 'type' in scatter_data:
            if isinstance(scatter_data["type"], int):
                scatter_fig.line.color = px.colors.qualitative.Light24[scatter_data["type"]]
            desc += f'type: {scatter_data["type"]}'
        if 'state' in scatter_data:
            desc += f'\nstates: {scatter_data["state"]}'
        if len(desc) > 0:
            scatter_fig.text = desc
        if 'marker_size' in scatter_data:
            scatter_fig.marker.size = scatter_data['marker_size']
        if scatter_data['mode'] == 'markers':
            scatter_fig.marker.line.width = 2
            scatter_fig.marker.line.color = '#ffffff'
        if 'fill' in scatter_data:
            scatter_fig.fill = 'toself'

        return scatter_fig
