import numpy as np
import plotly.express as px
import plotly.graph_objs as go

from dash_viz.data import Point3D
from debug_utils import timing


class CustomFigure:
    figure: go.Figure = go.Figure()

    def __init__(self, title: str = ''):
        self.__plot_data = None
        self.__init_base_layout(title)
        self.__updatable = True
        self.__scene_limits = Point3D()

    def __init_base_layout(self, title: str) -> None:
        self.figure.layout = {}
        self.figure.update_layout(
            title=dict(
                x=0.5, y=0.98, font=dict(family="Arial", size=30, color='#ffffff'),
                text=f'<b>{title}</b>', xanchor='center', yanchor='top'
            ),
            margin=dict(l=10, r=10, b=10, t=55, pad=0),
            template="plotly_dark",
            uirevision=True)

    @timing
    def update(self, plot_data: dict) -> go.Figure:
        self.__plot_data = plot_data.copy()
        if self.__updatable:
            self.__updatable = False
            if len(self.figure.data) > 0:
                self.figure.data = []
            self.__update_data()
            self.__update_layout()
            self.__updatable = True
        return self.figure

    def __update_data(self) -> None:
        scene_limits = Point3D()
        for scatter_name, scatter_data in self.__plot_data['scatters'].items():
            if self.__plot_data['scene_centric_data']:
                scene_limits.x = max(scene_limits.x, np.abs(scatter_data['x']).max())
                scene_limits.y = max(scene_limits.y, np.abs(scatter_data['y']).max())
                if self.__plot_data['type'] == '3D':
                    scene_limits.z = max(scene_limits.z, np.abs(scatter_data['z']).max())
            scatter_fig = self.create_scatter(self.__plot_data['type'], scatter_data, scatter_name)
            self.figure.add_trace(scatter_fig)
        if self.__plot_data['scene_centric_data']:
            self.__scene_limits = scene_limits

    def __update_layout(self) -> dict:
        layout = dict(
            title=dict(
                x=0.5, y=0.98, font=dict(family="Arial", size=30, color='#ffffff'),
                text=f'<b>{self.__plot_data["title"]}</b>', xanchor='center', yanchor='top'
            ),
            scene=dict(
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis'),
                zaxis=dict(title='Z Axis'),
                aspectmode='data'
            )
        )
        if self.__plot_data['type'] == '2D':
            self.figure.update_xaxes(scaleanchor='y')

        if 'scene_camera' in self.__plot_data:
            layout['scene_camera'] = self.__plot_data['scene_camera']

        if self.__plot_data['scene_centric_data']:
            layout['scene']['xaxis'] = dict(title='X Axis', range=[-self.__scene_limits.x, self.__scene_limits.x])
            layout['scene']['yaxis'] = dict(title='Y Axis', range=[-self.__scene_limits.y, self.__scene_limits.y])
            layout['scene']['zaxis'] = dict(title='Z Axis', range=[-self.__scene_limits.z, self.__scene_limits.z])
        self.figure.update_layout(layout)
        return layout

    @staticmethod
    def create_scatter(plot_type: str, scatter_data: dict, scatter_name: str) -> go.Scatter | go.Scatter3d:
        if plot_type == '3D':
            scatter_fig = go.Scatter3d(
                x=np.array(scatter_data['x']), y=np.array(scatter_data['y']), z=np.array(scatter_data['z']))
        else:
            scatter_fig = go.Scattergl(
                x=np.array(scatter_data['x']), y=np.array(scatter_data['y']))

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
