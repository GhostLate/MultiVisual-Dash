import numpy as np
import tensorflow as tf

from dataloaders.utils import rotate_bbox


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


def get_bbox_scatters(data):
    scatters = []
    for bbox_label in data['bbox_labels']:
        agent_center_x = bbox_label.box.center_x
        agent_center_y = bbox_label.box.center_y
        agent_center_z = bbox_label.box.center_z
        agent_rect_height = bbox_label.box.height
        agent_rect_length = bbox_label.box.length
        agent_rect_width = bbox_label.box.width
        agent_name = bbox_label.id
        agent_type = bbox_label.type

        box = rotate_bbox(agent_center_x, agent_center_y, agent_center_z,
                          agent_rect_length, agent_rect_width, agent_rect_height,
                          bbox_label.box.heading, 0, 0)

        scatter = {
            'name': f'agent_{agent_name}',
            'mode': 'lines',
            'x': box[0, :],
            'y': box[1, :],
            'z': box[2, :],
            'line_size': 4,
            # 'fill': True,
            'type': agent_type
        }
        scatters.append(scatter)
    return scatters


def get_point_scatters(data):
    points_all = data['points_all']
    point_labels_all = data['point_labels_all']

    types = np.unique(point_labels_all[:, 1])
    scatters = []
    for type in types:
        ids = np.where(point_labels_all == type)[0]
        x = points_all[ids, 0]
        y = points_all[ids, 1]
        z = points_all[ids, 2]
        scatter = {
            'name': f'points_{type}',
            'mode': 'markers',
            'x': x,
            'y': y,
            'z': z,
            'type': type,
            'marker_size': 2
        }
        scatters.append(scatter)
    return scatters
