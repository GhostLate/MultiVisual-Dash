from multi_visual_dash.dash_viz.data import DashMessage, ScatterData
from multi_visual_dash.dataloaders.dynamic_objects_dataset.dataloader import DynamicObjectsDataLoader
from multi_visual_dash.dataloaders.utils import bbox2plotly_drawing


class DynamicObjectsVisualizer:
    def __init__(self, dataset_dir: str, min_pc_points: int, max_dist2bbox: int, visualize_bboxes: bool = False):
        self.data_dir = dataset_dir
        self.dataloader = DynamicObjectsDataLoader(dataset_dir)
        self.min_pc_points = min_pc_points
        self.max_dist2bbox = max_dist2bbox
        self.visualize_bboxes = visualize_bboxes

    def __call__(self):
        for scenes in self.dataloader():
            f_scenes = self.dataloader.filter_scenes(scenes, self.min_pc_points, self.max_dist2bbox)
            t_scenes = self.dataloader.transform_scenes(f_scenes)
            c_scenes = self.dataloader.center_scenes(t_scenes)
            for scene_name, scene_data in f_scenes.items():
                for sample_id, sample_data in scene_data['samples'].items():
                    viz_massage = self.sample_data2plot_data(f'{sample_id}_f', sample_data)
                    yield viz_massage
                    viz_massage = self.sample_data2plot_data(f'{sample_id}_t', t_scenes[scene_name]['samples'][sample_id])
                    yield viz_massage
                    viz_massage = self.sample_data2plot_data(f'{sample_id}_c', c_scenes[scene_name]['samples'][sample_id])
                    yield viz_massage

    def sample_data2plot_data(self, sample_id, sample_data):
        viz_massage = DashMessage('add_plot', sample_id, False)
        for timestamp, ts_data in sample_data['timestamps'].items():
            for point_cloud in ts_data['points_clouds']:
                scatter = ScatterData(
                    name=f"ps_{timestamp}_{point_cloud['label_id']}",
                    mode='markers',
                    x=point_cloud['points'][:, 0], y=point_cloud['points'][:, 1], z=point_cloud['points'][:, 2])
                scatter.marker_size = 2
                viz_massage.scatters.append(scatter)
            if self.visualize_bboxes:
                bbox = ts_data['bbox']['original_points']
                bbox = bbox2plotly_drawing(bbox)
                scatter = ScatterData(
                    name=f"bbox_{timestamp}_{ts_data['bbox']['label_id']}",
                    mode='lines',
                    x=bbox[:, 0], y=bbox[:, 1], z=bbox[:, 2])
                scatter.line_size = 4
                viz_massage.scatters.append(scatter)
        return viz_massage
