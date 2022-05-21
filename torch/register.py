class registy:
    __instance = dict()
    def __new__(cls, *args):
        if args[0].lower() not in cls.__instance:
            cls.__instance[args[0].lower()] = object.__new__(cls)
        return cls.__instance[args[0].lower()]

    def __init__(self, introduction, *args):
        self.introduction = introduction
        self.__moduleDict = dict()

    def __str__(self):
        return self.introduction

    def register(self, cls):
        moduleName = cls.__name__
        if moduleName not in self.__moduleDict:
            self.__moduleDict[moduleName] = cls
        return cls

    def get_module_list(self):
        return list(self.__moduleDict.keys())

    def get_module(self, name):
        if name in self.__moduleDict:
            return self.__moduleDict[name]
        else:
            raise KeyError(f'{name} is not found in the registy.[Tips : You can use get_module_list function to show the registered module]')
