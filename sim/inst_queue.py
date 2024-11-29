from tools import *
from collections import deque

class inst_group:
    def __init__(self, id, pre_group, command_queue = []):
        self.id = id
        self.pre_group = pre_group
        self.queue = deque(command_queue)

    def issuable(self, group_list):
        if not set(group_list).intersection(set(self.pre_group)):
            return True
        else:
            return False
        
    def is_empty(self):
        if self.queue: return False
        else: return True

    def get_inst(self):
        assert not self.is_empty()
        return self.queue[0]
    
    def issue_inst(self):
        assert not self.is_empty()
        self.queue.popleft()

class inst_queue:
    def __init__(self):
        self.groups = {}

    def add_group(self, id, pre_group, command_queue = []):
        assert id not in self.groups.keys()
        self.groups[id] = inst_group(id, pre_group, command_queue)

    # return the issuable group
    def issuable_group(self):
        all_group = self.groups.keys()
        issuable_group = []
        for group in all_group:
            if self.groups[group].issuable(all_group):
                issuable_group.append(group)
        return issuable_group

    def get_inst(self):
        issuable_group = self.issuable_group()
        inst_list = []
        for group in issuable_group:
            inst = self.groups[group].get_inst()
            inst_list.append(
                (group, inst)
            )
        return inst_list

    def issue_inst(self, group_id):
        self.groups[group_id].issue_inst()

    def clear_empty_group(self, group_id):
        if self.groups[group_id].is_empty():
            self.groups.pop(group_id)

    def check_empty(self):
        return len(self.groups) == 0