from multi_visual_dash.dash_viz.data import DashMessage
from multi_visual_dash.dataloaders.waymo.motion.dataloader import WaymoMotionDataLoader
from multi_visual_dash.dataloaders.waymo.motion.utils import get_road_scatters, get_car_rect_scatters, \
    get_trajectory_scatters, get_light_scatters, get_pred_trajectory_scatters


class WaymoMotionVisualizer:
    def __init__(self, tfrecord_dir: str, save_dir: str = None, center_data: bool = True):
        self.save_dir = save_dir
        self.center_data = center_data
        self.waymo_perception_dataloader = WaymoMotionDataLoader(tfrecord_dir)

    def __call__(self):
        for idx, raw_data in enumerate(self.waymo_perception_dataloader()):
            yield self.data2plot_data(raw_data)

    def data2plot_data(self, data: dict, pred_traj_data: dict = None) -> DashMessage:
        if self.center_data:
            viz_massage = DashMessage('new_plot', str(data['scenario/id'].numpy().astype(str)[0]))
        else:
            viz_massage = DashMessage('add2plot', 'main')

        if self.save_dir is not None:
            viz_massage.save_dir = self.save_dir

        road_scatters = get_road_scatters(data)
        viz_massage.scatters.extend(road_scatters)
        light_scatters = get_light_scatters(data)
        viz_massage.scatters.extend(light_scatters)
        car_rect_scatters = get_car_rect_scatters(data)
        viz_massage.scatters.extend(car_rect_scatters)

        if pred_traj_data is not None:
            trajectory_scatters = get_trajectory_scatters(data, [pred_traj_data['agent_id']])
            viz_massage.scatters.extend(trajectory_scatters)
            trajectory_scatters = get_pred_trajectory_scatters(
                data, pred_traj_data['coords'], pred_traj_data['probas'], pred_traj_data['agent_id'])
            viz_massage.scatters.extend(trajectory_scatters)
        else:
            trajectory_scatters = get_trajectory_scatters(data)
            viz_massage.scatters.extend(trajectory_scatters)

        return viz_massage

    def get_plot_data_with_pred(self, scenario_id: str, pred_traj_data: dict, tfrecord_name: str) -> DashMessage:
        """
            :param scenario_id: as is
            :param pred_traj_data: {
                'coords': np.ndarray [samples; points; x, y],
                'probas': np.ndarray [samples],
                'agent_id': int
            }
            :param tfrecord_name: as is
            :return: plot_data: DashMessage
        """
        data = self.waymo_perception_dataloader.get_data_by_scenario_id(scenario_id, tfrecord_name)
        return self.data2plot_data(data, pred_traj_data)
