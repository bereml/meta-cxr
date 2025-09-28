""" backbones.py """


class Registry(dict):

    def register(self, name):
        def decorator_register(obj):
            self[name] = obj
            return obj
        return decorator_register


BACKBONES = Registry()


__all__ = ['BACKBONES']
