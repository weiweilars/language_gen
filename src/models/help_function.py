from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import pydoc
import torch
import re
import pdb
from collections import OrderedDict
from functools import reduce

EPSILON = 1e-15

def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)

def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result

def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def mean_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)
        
    result = torch.tensor(0)
    for i in range(num_class):
        if i != 0:
            temp_output = (logits == i).int()
            temp_target = (targets == i).int()

            intersection = (targets * output).sum()

            union = targets.sum() + output.sum() - intersection

            result += (intersection + EPSILON) / (union + EPSILON)

    result = result/(num_class-1)

    return result

def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()

def complete_info(class_info):

    #ordered_class_info = list(OrderedDict(sorted(class_info['seg_classes'].items())).items())
    
    ordered_class_info = [(name, class_info['seg_classes'][name]) for name in class_info['seg_classes']['heads_order']]

    max_seg_dim = max([len(i[1]) for i in ordered_class_info])
    classes = [d[1] for d in ordered_class_info]

    if class_info['seg_classes']['split_type'] == 1:
        # [[9], [5], [7, 32, 4]]
        total_seg_heads = [list(filter(lambda a: a != 0, i[1])) for i in ordered_class_info if sum(i[1]) != 0]

    elif class_info['seg_classes']['split_type']  == 0:
        # [[9, 5, 7, 32, 4]]
        total_seg_heads = [[i for i in reduce(lambda x,y:x+y,classes) if i !=0]]

    elif class_info['seg_classes']['split_type']  == 2:
        # [[9], [5], [7], [32], [4]]
        total_seg_heads = [[i] for i in reduce(lambda x,y:x+y,classes) if i != 0]

    else:
        # try some customer made combination
        pass

    num_classes = 0
 
    if 'class_label' in class_info:

        class_label = {'other': 0}
        idx = 1
        for i in class_info['class_label']:

            name = i.split('_')[0]

            if sum(class_info['seg_classes'][name]) != 0:

                class_label[i] = idx

                idx += 1
            else:
                class_label[i] = -1
            
        
        num_classes = idx

    else:
        class_label = {key:-1 for key in class_info['class_label']}

    if not class_info['add_classification']:
        num_classes = 0


    # add_classification = class_info['add_classification']
    # if add_classification:
    #     class_label = {}
    #     idx = 0
    #     class_list = []
    #     for ci in ordered_class_info:
    #         if sum(ci[1]) != 0:
    #             class_label[ci[0]] = idx
    #             class_list.append(ci[0])
    #             idx += 1
                
    #     if idx > 1:
    #         aux_params['classes'] = idx
    #     else:
    #         aux_params['classes'] = 2
    #         class_label[class_list[0]] = 1

    # else:
    #     class_label = {key:0 for key,_ in class_info['seg_classes'].items()}

    return ordered_class_info, total_seg_heads, num_classes, class_label, max_seg_dim
                
