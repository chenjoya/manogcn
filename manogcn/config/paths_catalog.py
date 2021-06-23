from os.path import join


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "freihand_train":{
            "image_dir": "freihand/training/rgb",    
            "K_file": "freihand/annotations/training_K.json",
            "vert_file": "freihand/annotations/training_verts.json",
        },
        "ho3d_train":{
            "root_dir": "ho3d",
            "image_file": "ho3d/annotations/training_images.json",
            "vert_file": "ho3d/annotations/training_verts.json",
            "K_file": "ho3d/annotations/training_Ks.json",
        },
        "freihand_test":{
            "image_dir": "freihand/evaluation/rgb",
            "K_file": "freihand/annotations/evaluation_K.json",
        },
        "ho3d_test":{
            "root_dir": "ho3d",
            "image_file": "ho3d/annotations/evaluation_images.json",
            "K_file": "ho3d/annotations/evaluation_Ks.json",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        if "freihand_train" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                image_dir=join(data_dir, attrs["image_dir"]),
                K_file=join(data_dir, attrs["K_file"]),
                vert_file=join(data_dir, attrs["vert_file"]),
            )
            return dict(
                factory="FreiHAND",
                args=args,
            )
        if "ho3d_train" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root_dir = join(data_dir, attrs["root_dir"]),
                image_file=join(data_dir, attrs["image_file"]),
                K_file=join(data_dir, attrs["K_file"]),
                vert_file=join(data_dir, attrs["vert_file"]),
            )
            return dict(
                factory="HO3D",
                args=args,
            )
        if "freihand_test" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                image_dir=join(data_dir, attrs["image_dir"]),
                K_file=join(data_dir, attrs["K_file"]),
            )
            return dict(
                factory="FreiHAND",
                args=args,
            )
        if "ho3d_test" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root_dir = join(data_dir, attrs["root_dir"]),
                image_file=join(data_dir, attrs["image_file"]),
                K_file=join(data_dir, attrs["K_file"]),
            )
            return dict(
                factory="HO3D",
                args=args,
            )
        else:
            assert False, "Unknown Dataset."
        
