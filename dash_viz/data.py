import numpy as np


class Point3D:
    def __init__(self, x: int | float = 0, y: int | float = 0, z: int | float = 0):
        self.x, self.y, self.z = x, y, z


class SceneCamera:
    def __init__(self):
        self.up = Point3D()
        self.center = Point3D()
        self.eye = Point3D()

    def __iter__(self):
        for key in self.__dict__:
            yield key, getattr(self, key).__dict__

    def to_dict(self):
        return dict(self)


class ScatterData:
    line_size: int | float
    line_type: int | float
    type: int | str
    desc: int | float | str
    marker_size: int | float
    marker_line_width: int | float
    fill: bool
    opacity: int | float

    def __init__(self, name: str, mode: str, x: list | np.ndarray, y: list | np.ndarray, z: list | np.ndarray = None):
        self.mode = mode
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        for key in self.__dict__:
            if getattr(self, key) is not None:
                yield key, getattr(self, key)

    def to_dict(self):
        return dict(self)


class DashMessage:
    scene_camera: SceneCamera
    save_dir: str
    title: str

    def __init__(self, command_type: str, plot_name: str, scene_centric_data=False):
        self.command_type = command_type
        self.plot_name = plot_name
        self.scene_centric_data = scene_centric_data
        self.scatters = list()

    def __iter__(self):
        for key in self.__dict__:
            if key == 'scatters' and isinstance(getattr(self, key), list):
                yield key, [dict(i) for i in getattr(self, key) if isinstance(i, ScatterData)]
            elif isinstance(getattr(self, key), SceneCamera):
                yield key, dict(getattr(self, key))
            else:
                yield key, getattr(self, key)

    def to_dict(self):
        return dict(self)
