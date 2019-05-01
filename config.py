###################### load packages ####################
import os
import configparser


###################### config 类 ####################
class ConfigHandler(object):
    """
    Main class of configuration handler
    """
    def __init__(self, path: str):
        """
        Load config file
        :param path: config file
        """
        self.path = self.__load_path(path)
        self.config = dict()
        if os.path.exists(self.path) is False:
            raise IOError("%s not exists" % self.path)
        self.cf = configparser.ConfigParser()
        self.cf.read(self.path)

    @staticmethod
    def __load_path(path: str):
        if os.path.isabs(path) is True:
            return path
        else:
            return os.path.realpath(path)

    def load_config(self):
        """
        Load ini file into dict
        """
        for sec in self.cf.sections():
            self.config[sec] = dict()
            for opt in self.cf.options(sec):
                self.config[sec][opt] = self.cf.get(sec, opt)

    def export_config(self):
        """
        Export modified config into ini file.
        """
        for sec in self.config:
            if self.cf.has_section(sec) is False:
                self.cf.add_section(sec)
            for opt in self.config[sec]:
                self.cf.set(sec, opt, self.config[sec][opt])
        with open(self.path, "w") as fp:
            self.cf.write(fp)