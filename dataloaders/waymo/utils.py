import numpy as np
import scipy
from sklearn.preprocessing import minmax_scale


def get_slices(samples_id) -> list:
    start_val = samples_id[0]
    start_id = 0
    vals = []
    for idx, val in enumerate(samples_id[1:]):
        if val != start_val:
            vals.append([start_val, start_id, idx + 1])
            start_val = val
            start_id = idx + 1
    vals.append([start_val, start_id, samples_id.shape[0]])
    return vals


def remove_unvalid_data(valid: np.array, data: np.array) -> np.array:
    return data[np.where(valid == 1)]


def filter_valid_data(valid: np.array, data: np.array) -> list:
    data = data.tolist()
    indexes = np.where(valid == 0)[0].tolist()
    for i in indexes:
        data[i] = None
    return data


def get_road_scatters(data: dict) -> list:
    roads_type = data['roadgraph_samples/type'].numpy().squeeze()
    roads_valid = data['roadgraph_samples/valid'].numpy().squeeze()
    roads_xyz = data['roadgraph_samples/xyz'].numpy()

    ids_slices = get_slices(data['roadgraph_samples/id'].numpy().squeeze())
    scatters = []
    for ids_slice in ids_slices:
        if ids_slice[0] >= 0:
            x = filter_valid_data(roads_valid[ids_slice[1]: ids_slice[2]], roads_xyz[ids_slice[1]: ids_slice[2], 0])
            y = filter_valid_data(roads_valid[ids_slice[1]: ids_slice[2]], roads_xyz[ids_slice[1]: ids_slice[2], 1])
            scatter = {
                'name': f'road_line_{ids_slice[0]}',
                'mode': 'lines',
                'x': x,
                'y': y,
                'type': roads_type[ids_slice[1]],
                'line_size': 1,
            }
            if roads_type[ids_slice[1]] == 2:
                scatter['line_type'] = 'dash'
            scatters.append(scatter)
    return scatters


def get_light_scatters(data: dict) -> list:
    lights_state = np.vstack(
        [data['traffic_light_state/past/state'].numpy(), data['traffic_light_state/current/state'].numpy()])
    lights_valid = data['traffic_light_state/current/valid'].numpy().squeeze()
    lights_x = data['traffic_light_state/current/x'].numpy().squeeze()
    lights_y = data['traffic_light_state/current/y'].numpy().squeeze()
    scatters = []
    for idx in np.where(lights_valid == 1)[0]:
        scatter = {
            'name': f'lights_{idx}',
            'mode': 'markers',
            'x': [lights_x[idx]],
            'y': [lights_y[idx]],
            'type': lights_state[-1, idx],
            'desc': f'states: {lights_state[:, idx]}',
            'marker_size': 10
        }
        scatters.append(scatter)
    return scatters


def get_car_rect_scatters(data: dict) -> list:
    agent_rect_x = data['state/current/x'].numpy().squeeze()
    agent_rect_y = data['state/current/y'].numpy().squeeze()
    agent_rect_z = data['state/current/z'].numpy().squeeze()
    agent_rect_height = data['state/current/height'].numpy().squeeze()
    agent_rect_length = data['state/current/length'].numpy().squeeze()
    agent_rect_width = data['state/current/width'].numpy().squeeze()
    agent_rect_bbox_yaw = data['state/current/bbox_yaw'].numpy().squeeze()
    agent_valid = data['state/current/valid'].numpy().squeeze()
    agent_id = data['state/id'].numpy().astype(int).squeeze()
    agent_type = data['state/type'].numpy().astype(int).squeeze()
    scatters = []
    for idx in np.where(agent_valid == 1)[0]:
        xyz = np.array([agent_rect_x[idx], agent_rect_y[idx], agent_rect_z[idx]]).reshape(-1, 1)
        yaw = agent_rect_bbox_yaw[idx]

        r_y = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        r_b = np.array([[np.cos(0), 0, np.sin(0)],
                        [0, 1, 0],
                        [-np.sin(0), 0, np.cos(0)]])
        r_a = np.array([[1, 0, 0],
                        [0, np.cos(0), -np.sin(0)],
                        [0, np.sin(0), np.cos(0)]])

        l, w, h = agent_rect_length[idx] / 2, agent_rect_width[idx] / 2, agent_rect_height[idx] / 2
        box_up = np.array([[l, l, -l, -l],
                           [-w, w, w, -w],
                           [h, h, h, h]])
        box_down = np.array([[l, l, -l, -l],
                             [-w, w, w, -w],
                             [-h, -h, -h, -h]])
        box_up = (r_y @ r_b @ r_a) @ box_up
        box_up = xyz + np.append(box_up, box_up[:, 0].reshape(-1, 1), axis=1)
        box_down = (r_y @ r_b @ r_a) @ box_down
        box_down = xyz + np.append(box_down, box_down[:, 0].reshape(-1, 1), axis=1)

        box = np.hstack([box_down, box_up])
        scatter = {
            'name': f'agent_{agent_id[idx]}',
            'mode': 'lines',
            'x': box[0, :5],
            'y': box[1, :5],
            'line_size': 1,
            'fill': True,
            'type': agent_type[idx]
        }
        scatters.append(scatter)
    return scatters


def get_trajectory_scatters(data: dict, show_f_traj_ids: list = None) -> list:
    agent_traj_x = np.hstack([data['state/past/x'].numpy(), data['state/current/x'].numpy()])
    agent_traj_y = np.hstack([data['state/past/y'].numpy(), data['state/current/y'].numpy()])
    agent_traj_future_x = data['state/future/x'].numpy()
    agent_traj_future_y = data['state/future/y'].numpy()
    agent_valid = np.hstack([data['state/past/valid'].numpy(), data['state/current/valid'].numpy()])
    agent_future_valid = data['state/future/valid'].numpy()
    agent_id = data['state/id'].numpy().astype(int).squeeze()
    agent_type = data['state/type'].numpy().astype(int).squeeze()
    scatters = []
    for idx in np.where(agent_valid[:, -1] == 1)[0]:
        idx_valid = np.where(agent_valid[idx] == 1)[0]
        min_idx, max_idx = np.min(idx_valid), np.max(idx_valid) + 1
        x = filter_valid_data(agent_valid[idx, min_idx: max_idx], agent_traj_x[idx, min_idx: max_idx])
        y = filter_valid_data(agent_valid[idx, min_idx: max_idx], agent_traj_y[idx, min_idx: max_idx])
        scatter = {
            'name': f'traj_{agent_id[idx]}',
            'mode': 'lines+markers',
            'x': x,
            'y': y,
            'type': agent_type[idx],
            'marker_size': 4,
            'line_size': 2,
            # 'line_type': 'dash'
        }
        scatters.append(scatter)

        idx_valid = np.where(agent_future_valid[idx] == 1)[0]
        if len(idx_valid) > 0 and (show_f_traj_ids is None or agent_id[idx] in show_f_traj_ids):
            min_idx, max_idx = np.min(idx_valid), np.max(idx_valid) + 1
            x = filter_valid_data(agent_future_valid[idx, min_idx: max_idx], agent_traj_future_x[idx, min_idx: max_idx])
            y = filter_valid_data(agent_future_valid[idx, min_idx: max_idx], agent_traj_future_y[idx, min_idx: max_idx])
            scatter = {
                'name': f'f_traj_{agent_id[idx]}',
                'mode': 'lines+markers',
                'x': x,
                'y': y,
                'type': agent_type[idx] + 3,
                'marker_size': 4,
                'line_size': 2,
                # 'line_type': 'dot'
            }
            scatters.append(scatter)
    return scatters


def get_pred_trajectory_scatters(data: dict, coords: np.array, probas: np.array, agent_id: int) -> list:
    scatters = []
    agents_type = data['state/type'].numpy().astype(int).squeeze()
    agents_id = data['state/id'].numpy().astype(int).squeeze()
    agent_type = agents_type[np.where(agents_id == agent_id)[0]][0]
    probas = scipy.special.softmax(probas)
    opacities = minmax_scale(probas, feature_range=(0.3, 0.999))

    for idx in range(coords.shape[0]):
        scatter = {
            'name': f'p_traj_{agent_id}/{idx}',
            'mode': 'lines+markers',
            'x': coords[idx, :, 0],
            'y': coords[idx, :, 1],
            'type': agent_type + 4,
            'marker_size': 2,
            'line_size': 1,
            'opacity': opacities[idx],
            'desc': f'probability: {probas[idx]}',
            # 'line_type': 'dash'
        }
        scatters.append(scatter)
    return scatters