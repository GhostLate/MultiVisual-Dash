import numpy as np
import tensorflow as tf

from multi_visual_dash.dash_viz.data import ScatterData
from multi_visual_dash.dataloaders.utils import transform_points, get_bbox, bbox2plotly_drawing

def range_image2pc_labels(frame, range_images, segmentation_labels, ri_index=0):
    """
    Convert segmentation labels from range images to point clouds.
    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.
    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """

    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def get_bbox_scatters(data: dict, center_data: bool = False) -> list[ScatterData]:
    scatters = []
    for bbox_label in data['bboxes']:
        bbox = get_bbox(bbox_label['center'][0], bbox_label['center'][1], bbox_label['center'][2],
                        bbox_label['extent'][0], bbox_label['extent'][1], bbox_label['extent'][2],
                        bbox_label['heading'], 0, 0)

        bbox = bbox2plotly_drawing(bbox)
        if not center_data:
            bbox = transform_points(bbox, data['transform_matrix'])
        scatter = ScatterData(
            name=f"agent_{bbox_label['name']}",
            mode='lines',
            x=bbox[:, 0], y=bbox[:, 1], z=bbox[:, 2])
        scatter.line_size = 4
        scatter.type = bbox_label['type']
        scatters.append(scatter)
    return scatters


def get_point_scatters(data: dict, center_data: bool = False) -> list[ScatterData]:
    points_all = data['points_all']
    point_labels_all = data['point_labels_all']

    point_types = np.unique(point_labels_all[:, 1])
    scatters = []
    if not center_data:
        points_all = transform_points(points_all, data['transform_matrix'])
    for point_type in point_types:
        if not center_data and point_type not in [8, 9, 10, 11, 14, 15, 16, 17, 19]:
            continue
        ids = np.where(point_labels_all == point_type)[0]
        x = points_all[ids, 0]
        y = points_all[ids, 1]
        z = points_all[ids, 2]
        scatter = ScatterData(
            name=f'points_{point_type}',
            mode='markers',
            x=x, y=y, z=z)
        scatter.type = point_type
        scatter.marker_size = 1
        scatters.append(scatter)
    return scatters
