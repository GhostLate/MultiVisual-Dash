from multi_visual_dash.dash_viz.data import DashMessage
from multi_visual_dash.dataloaders.waymo.perception.dataloader import WaymoPerceptionDataLoader
from multi_visual_dash.dataloaders.waymo.perception.utils import get_point_scatters, get_bbox_scatters


class WaymoPerceptionVisualizer:
    def __init__(self, tfrecord_dir: str, save_dir: str = None, center_data: bool = True):
        self.save_dir = save_dir
        self.center_data = center_data
        self.waymo_perception_dataloader = WaymoPerceptionDataLoader(tfrecord_dir)

    def __call__(self):
        for idx, raw_data in enumerate(self.waymo_perception_dataloader()):
            yield self.data2plot_data(raw_data)

    def data2plot_data(self, raw_data: dict) -> DashMessage:
        if self.center_data:
            viz_massage = DashMessage('new_plot', raw_data['name'], True)
        else:
            viz_massage = DashMessage('add2plot', raw_data['name'], False)

        if self.save_dir is not None:
            viz_massage.save_dir = self.save_dir

        point_scatters = get_point_scatters(raw_data, self.center_data)
        viz_massage.scatters.extend(point_scatters)
        point_scatters = get_bbox_scatters(raw_data, self.center_data)
        viz_massage.scatters.extend(point_scatters)
        return viz_massage
