class Paw(object):
    def __init__(self, name, ground=False):
        self.global_pos = (-1, -1)
        self.ground = ground
        self.name = name


paw_obj1 = Paw('fl')
paw_obj2 = Paw('fr')
paw_obj3 = Paw('bl')
paws = [paw_obj3, paw_obj2, paw_obj1]
paw_dict = {'fl': -1, 'fr': 2, 'bl': 3, 'br': 10}
