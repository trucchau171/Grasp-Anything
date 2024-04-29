def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'grasp-anything':
        from .grasp_anything_data import GraspAnythingDataset
        return GraspAnythingDataset
    elif dataset_name == 'vmrd':
        from .vmrd_data import VMRDDataset
        return VMRDDataset
    elif dataset_name == 'ocid':
        from .ocid_grasp_data import OCIDGraspDataset
        return OCIDGraspDataset
    elif dataset_name == 'grasp-anything++':
        from .grasp_anything_plus_data import GraspAnythingPlusDataset
        return GraspAnythingPlusDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
