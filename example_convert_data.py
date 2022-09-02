from multi_visual_dash.dataloaders.waymo.perception.dynamic_objects_converter import DynamicObjetsConverter

if __name__ == "__main__":
    waymo_data_loader = DynamicObjetsConverter('./data/perception/', './data/dynamic_objects_dataset/waymo/')

