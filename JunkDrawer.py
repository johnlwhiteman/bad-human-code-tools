#!/usr/bin/env python3
import csv
import datetime
import hashlib
import json
import os
import pprint
import random
import shutil
import sys
import time
import xml.dom.minidom

class Msg:

    LABEL = "BadHumanCode"

    @staticmethod
    def abort(msg):
        Msg.raw("[{0}][Abort]: {1}".format(Msg.LABEL, msg), isErrorFlag=True)
        sys.exit(1)

    @staticmethod
    def error(msg, abortFlag=False):
        Msg.raw("[{0}][Error]: {1}".format(Msg.LABEL, msg), isErrorFlag=True)
        if abortFlag:
            sys.exit(1)

    @staticmethod
    def flush():
        sys.stdout.flush()
        sys.stderr.flush()

    @staticmethod
    def pretty(msg, formatType="json"):
        if formatType == "xml":
            msg = xml.dom.minidom.parseString(msg)
            Msg.raw(msg.toprettyxml(indent="  "))
        else:
            pprint.pprint(msg)
        Msg.flush()

    @staticmethod
    def raw(msg, isErrorFlag=False):
        if isErrorFlag:
            sys.stderr.write("{0}\n".format(msg))
            sys.stderr.flush()
        else:
            sys.stdout.write("{0}\n".format(msg))
            sys.stdout.flush()

    @staticmethod
    def show(msg):
        Msg.raw("[{0}]: {1}".format(Msg.LABEL, msg), isErrorFlag=False)


class Data:

    @staticmethod
    def appendListValues(aList, value):
        return ["{0}{1}".format(i, value ) for i in aList]

    @staticmethod
    def buildDictByKeys(keys, defaultValue=None):
        if isinstance(defaultValue, list) :
            return {key: list(defaultValue) for key in keys}
        return {key: defaultValue for key in keys}

    @staticmethod
    def getDictByKeys(aDict, keys, defaultValue=None):
        return {key: aDict.get(key, defaultValue) for key in keys}

    @staticmethod
    def getDictValuesByKeys(aDict, keys, defaultValue=None):
        return list(Data.getDictByKeys(aDict, keys, defaultValue).values())

    @staticmethod
    def prependListValues(aList, value):
        return ["{0}{1}".format(value, i) for i in aList]


class DateTime:

    @staticmethod
    def convertEpochToTimestamp(epoch):
        return time.strftime("%Y-%m-%dT%H:%M:%S+0000", time.localtime(epoch))

    @staticmethod
    def convertTimestampToEpoch(timestamp):
        return int(time.mktime(time.strptime(timestamp, '%Y-%m-%dT%H:%M:%S+0000')))

    @staticmethod
    def getEpoch():
        return int(time.time())

    @staticmethod
    def getTimestamp():
        return DateTime.convertEpochToTimestamp(DateTime.getEpoch())


class Dir:

    @staticmethod
    def delete(path):
        if Dir.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def expandPath(path):
        return os.path.expanduser(path)

    @staticmethod
    def exists(path):
        return os.path.isdir(path)

    @staticmethod
    def getFiles(path, recursiveFlag=False, excludeDirFlag=False):
        paths = []
        if recursiveFlag:
            for rootPath, dirPaths, filePaths in os.walk(path):
                if excludeDirFlag:
                    paths.extend(filePaths)
                else:
                    paths.extend([os.path.join(rootPath, filePath) for filePath in filePaths])
                    for subDirPath in dirPaths:
                        dirPath = os.path.join(rootPath, subDirPath)
                        Dir.getFiles(dirPath, recursiveFlag, excludeDirFlag)
        else:
            for filePath in os.listdir(path):
                dirFilePath = os.path.join(path, filePath).replace('\\', '/')
                if os.path.isfile(dirFilePath):
                    if excludeDirFlag:
                        paths.append(filePath)
                    else:
                        paths.append(dirFilePath)
        if len(paths) < 1:
            return None
        return paths

    @staticmethod
    def getParent(path):
        return os.path.abspath(os.path.join(path, os.pardir))

    @staticmethod
    def make(path):
        if not Dir.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def trimLeft(path, subPathName):
        tokens = path.split("/")
        try:
            idx = tokens.index(subPathName)
            return "/".join(tokens[idx:]).strip()
        except ValueError:
            pass
        return path


class File:

    @staticmethod
    def copy(srcPath, tgtPath):
        try:
            shutil.copyfile(srcPath, tgtPath)
        except IOError as e:
            Msg.abort("Can't copy log file: {0} to {1}\n{2}".format(srcPath, tgtPath, e), True)

    @staticmethod
    def delete(path):
        if not os.path.isfile(path):
            return
        os.unlink(path)

    @staticmethod
    def exist(paths):
        for path in paths:
            if not File.exists(path):
                return False
        return True

    @staticmethod
    def exists(path):
        return os.path.isfile(path)

    @staticmethod
    def expandPath(path):
        return os.path.expanduser(path)

    @staticmethod
    def getAbsPath(path):
        return File.getCanonicalPath(os.path.abspath(path))

    @staticmethod
    def getBasename(path):
        return os.path.basename(path)

    @staticmethod
    def getCanonicalPath(path):
        return path.replace('\\', '/')

    @staticmethod
    def getDirectory(path):
        return File.getCanonicalPath(os.path.abspath(os.path.join(path, os.pardir)))

    @staticmethod
    def getExtension(path):
        return os.path.splitext(path)[1].strip().replace(".", "")

    @staticmethod
    def getFileSizeAsBytes(path):
        return os.path.getsize(path)

    @staticmethod
    def getModifiedTimeAsEpoch(path):
        return int(os.path.getmtime(path))

    @staticmethod
    def getNameOnly(path):
        return File.getCanonicalPath(os.path.basename(os.path.splitext(path)[0]))

    @staticmethod
    def read(path, asCsvFlag=False, asJsonFlag=False):
        content = None
        try:
            with open(path, "r",  encoding="utf-8") as fd:
                if asJsonFlag:
                    content = json.load(fd)
                elif asCsvFlag:
                    content = []
                    rows = csv.reader(fd, delimiter=",")
                    for row in rows:
                        content.append(row)
                else:
                    content = fd.read()
        except IOError as e:
            Msg.abort("Can't read log file: {0}\n{1}".format(path, e))
        return content

    @staticmethod
    def write(path, content, asJsonFlag=False):
        try:
            Dir.make(File.getDirectory(path))
            with open(path, "w", encoding="utf-8") as fd:
                if asJsonFlag:
                    json.dump(content, fd, indent=4, sort_keys=True)
                else:
                    fd.write(content)
        except IOError as e:
            Msg.abort("Can't write log file: {0}\n{1}".format(path, e))


class Number:

    @staticmethod
    def formatIntWithCommas(value):
        return str("{:,}".format(value))

    @staticmethod
    def asFloat(value, decimalPlaces=2):
        return float("{0:.{1}f}".format(value, decimalPlaces))

    @staticmethod
    def getRandomIndex(lst):
        return random.randint(0, len(lst) - 1)

    @staticmethod
    def getRandomNumber(low, high):
        return random.randint(low, high)

