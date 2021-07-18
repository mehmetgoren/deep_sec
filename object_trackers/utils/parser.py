import os
import yaml
from addict import Dict as dict


class YamlParser(dict):
    """
    This is yaml parser based on addict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(
        config_file='/mnt/sdc1/deep_sec/object_trackers/configs/deep_sort.yaml'):
    return YamlParser(config_file=config_file)

# if __name__ == "__main__":
#     cfg = YamlParser(config_file='../configs/deep_sort.yaml')
#     print(cfg)

# import ipdb
# ipdb.set_trace()
