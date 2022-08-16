import plotly.express as px
import plotly.graph_objs as go
from multiprocessing import Pool
from utils import timing


class CustomFigure:
    def __init__(self, plots_data: dict):
        self.plots_data = plots_data
        self._cur_plot = None
        self._cur_plot_type = None
        self.figure = go.Figure()
        self.__init_base_layout()
        self.__updatable = True

    def __init_base_layout(self) -> None:
        title = ''
        if self._cur_plot is not None:
            title = self.plots_data[self._cur_plot]['title']
        self.figure.layout = {}
        self.figure.update_layout(
            margin=dict(l=10, r=10, b=10, t=55, pad=0),
            title={'x': 0.5, 'y': 0.98,
                   'text': f'<b>{title}</b>',
                   'xanchor': 'center', 'yanchor': 'top',
                   'font': dict(family="Arial", size=30, color='#ffffff')},
            template="plotly_dark",
            uirevision=True)

    def update(self, cur_plot: str) -> go.Figure:
        if self.__updatable and self.plots_data and cur_plot in self.plots_data:
            self.__updatable = False
            if len(self.figure.data) > 0:
                self.figure.data = []
            self._cur_plot = cur_plot
            if self._cur_plot_type != self.plots_data[self._cur_plot]['type']:
                self._cur_plot_type = self.plots_data[self._cur_plot]['type']
                self.__update_layout()
            self.__update_data()
            self.__updatable = True
        return self.figure

    @timing
    def __update_data(self) -> None:
        for scatter_name, scatter_data in self.plots_data[self._cur_plot]['scatters'].copy().items():
            if all(key in scatter_data for key in ['x', 'y']):
                if self.plots_data[self._cur_plot]['type'] == '3D' and 'z' not in scatter_data:
                    scatter_data['z'] = [0] * len(scatter_data['x'])
                scatter_fig = create_scatter(self._cur_plot_type, scatter_data, scatter_name)
                self.figure.add_traces([scatter_fig])

    @timing
    def __update_data_pool(self) -> None:
        with Pool(processes=7) as pool:
            for scatter_name, scatter_data in self.plots_data[self._cur_plot]['scatters'].items():
                if all(key in scatter_data for key in ['x', 'y']):
                    if self.plots_data[self._cur_plot]['type'] == '3D' and 'z' not in scatter_data:
                        scatter_data['z'] = [0] * len(scatter_data['x'])
                    pool.apply_async(func=create_scatter, args=(self._cur_plot_type, scatter_data, scatter_name),
                                     callback=self.add_traces)
            pool.close()
            pool.join()

    def add_traces(self, scatter):
        self.figure.add_traces([scatter])

    def __update_layout(self) -> None:
        self.__init_base_layout()
        if self._cur_plot_type == '2D':
            self.figure.update_xaxes(scaleanchor='y')
            self.figure.update_layout(
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis'
                ))
        elif self._cur_plot_type == '3D':
            self.figure.update_layout(
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Z Axis',
                    aspectmode='data'
                ))


def create_scatter(plot_type: str, scatter_data: dict, scatter_name: str) -> go.Scatter | go.Scatter3d:
    if plot_type == '3D':
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
    if 'desc' in scatter_data:
        desc += f'\n{scatter_data["desc"]}'
    if len(desc) > 0:
        scatter_fig.text = desc
    if 'marker_size' in scatter_data:
        scatter_fig.marker.size = scatter_data['marker_size']
    if 'marker_line_width' in scatter_data:
        scatter_fig.marker.line.width = scatter_data['marker_line_width']
        scatter_fig.marker.line.color = '#ffffff'
    if 'fill' in scatter_data:
        scatter_fig.fill = 'toself'
    if 'opacity' in scatter_data and 0 >= scatter_data['opacity'] >= 1:
        scatter_fig.opacity = scatter_data['opacity']
    return scatter_fig
