import os

class subject(object):
    def __init__(self):
        self.image_path = None
        self.label_path = None
        self.ldmrk_path = None
        self.modality = None
        self.visit = None
        self.index = None
    
    def key(self):
        assert self.image_path is not None, 'image path must not be None'
        assert self.modality is not None, 'image modality must not be None'
        assert self.index is not None, 'image index must not be None'
        if self.visit is None:
            visit = 1
        return f'subject{self.index}-visit{visit}-{self.modality}'


class database(object):
    def __init__(self):
        self.subject_list = []

    def add(self, subejct):
        self.subject_list.append(subejct)

    def delete(self, delete_func):
        raise(NotImplementedError)

    def search(self, condition_func):
        pass

    def dump_data(self):
        pass

    def gen_all_pairs(self):
        pass
