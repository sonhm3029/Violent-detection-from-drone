import datetime

def generateUniquePrefix():
    currentTime = datetime.datetime.now()
    prefixDate = "_".join(str(currentTime.date()).split("-"))
    prefixTimestamp = "".join(str(currentTime.timestamp()).split("."))
    return f"{prefixDate}_{prefixTimestamp}"