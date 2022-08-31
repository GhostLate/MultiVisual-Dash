from multi_visual_dash.dash_viz.data import DashMessage, ScatterData
from multi_visual_dash.dataloaders.dynamic_objects_dataset.dataloader import DynamicObjectsDataLoader


class DynamicObjectsVisualizer:
    def __init__(self, dataset_dir: str, min_pc_points: int, max_dist2bbox: int):
        self.data_dir = dataset_dir
        self.dynamic_objects_dataloader = DynamicObjectsDataLoader(dataset_dir)
        self.post_data = self.dynamic_objects_dataloader.get_filtered_data(min_pc_points, max_dist2bbox)

    def __call__(self):
        for scene in self.post_data:
            for agent in self.post_data[scene]:
                viz_massage = DashMessage('add_plot', agent, False)
                for timestamp in self.post_data[scene][agent]['points']:
                    pc = self.post_data[scene][agent]['points'][timestamp][0]
                    pc_points = pc['points']
                    pc_type = pc['type']
                    scatter = ScatterData(
                        name=f'{timestamp}_{pc_type}',
                        mode='markers',
                        x=pc_points[:, 0], y=pc_points[:, 1], z=pc_points[:, 2])
                    scatter.marker_size = 2
                    viz_massage.scatters.append(scatter)
                yield viz_massage
