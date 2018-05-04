
import configparser

def readConfig(config_path, section, key):
	cp = configparser.ConfigParser()
	cp.read(config_path)
	val = cp.get(section, key)
	return val