import sys
import itertools
import random
from data import InputData, OutputData, Config
from typing import List 
from data import Edge

DEBUG = False

def Max(a, b):
    if a > b:
        return a
    else:
        return b

def computeCost(inputData:InputData, outputData:OutputData):
    areaMatchingCost: int = 0
    windowMatchingCost: int = 0
    for did in range(outputData.deviceNum):
        device = inputData.devices[did]
        rid = outputData.regionIndexs[did]
        
        # compute areaMatchingCost
        energyType = inputData.regions[rid].energyType
        installCost = device.energyCosts[energyType]
        areaMatchingCost += installCost

    # get the devices on core production line
    edgeOnCoreProductionLine: List[Edge] = []
    deviceOnCoreProductionLine: List[int] = []
    for eid in inputData.pipeline.edgeIndexs:
        edge = inputData.edges[eid]
        edgeOnCoreProductionLine.append(edge)
        # print("edge {} type {}".format(eid, edge.type))
    for edge in edgeOnCoreProductionLine:
        deviceOnCoreProductionLine.append(edge.sendDevice)
    deviceOnCoreProductionLine.append(edge.recvDevice)

    # find out window process time (device -> area -> energy -> processTime)
    windowProcessTimeRaw: List[List[int]] = [[0] for i in range(inputData.W)] # do not using * since list is not elementary type, * will make shallow copy
    for id, did in enumerate(deviceOnCoreProductionLine):
        wid = outputData.timeWindowIndexs[id]
        rid = outputData.regionIndexs[did]
        energyType = inputData.regions[rid].energyType
        processTime = inputData.energys[energyType].processTime
        windowProcessTimeRaw[wid].append(processTime)
    windowProcessTime = list(map(max, windowProcessTimeRaw))
    # compute entry time
    windowEntryTimes: List[int] = [0] * inputData.W
    for id, edge in enumerate(edgeOnCoreProductionLine):
        wid = outputData.timeWindowIndexs[id]
        wid_next = outputData.timeWindowIndexs[id + 1]
        windowEntryTimes[wid] += 1
        # if two consecutive windows are the same and the edge is red, i.e collabarative relation
        # only count once entry time
        if wid == wid_next and edge.type == 1:
            windowEntryTimes[wid] -= 1 # minus one since count one more time
            # print("collaboration in window {}".format(wid))
    # Don't forget the last device
    windowEntryTimes[wid_next] += 1

    windowProcessCost = 0
    windowPresetCost = 0
    for wid in range(inputData.W):
        entryTimes = windowEntryTimes[wid]
        processTime = windowProcessTime[wid]
        windowProcessCost += processTime * entryTimes
        windowPresetCost += processTime * inputData.windows[wid].costFactor
    windowProcessCost *= inputData.K
    windowMatchingCost = windowProcessCost + windowPresetCost
    totalCost = areaMatchingCost + windowMatchingCost
    if DEBUG:
        print('areaMatchingCost:\t{}'.format(areaMatchingCost))
        print('windowPresetCost:\t{}'.format(windowPresetCost))
        print('windowProcessCost:\t{}'.format(windowProcessCost))
        print('windowProcessTime:\t{}'.format(windowProcessTime))
        print('windowEntryTimes:\t{}'.format(windowEntryTimes))
        print('windowMatchingCost:\t{}'.format(windowMatchingCost))
    return totalCost

class WorkShop:
    def __init__(self, index):
        self.index = index
        self.minTi = Config.MAX_U32
        self.maxTi = 0
        self.anyRidOfEngine = [[] for i in range(Config.ENGINE_TYPE_NUM)]
        '''第i个元素储存支持第i种设备的area id(rid)，可以是列表（即多个满足条件的area）'''
        return


# Topological sort
class Queue:
    def __init__(self):
        self.vec = []
        self.headIndex = 0
        self.tailIndex = 0
        return

    def Push(self, item):
        self.vec.append(item)
        self.tailIndex = self.tailIndex + 1
        return

    def IsEmpty(self):
        return self.headIndex == self.tailIndex

    def Pop(self):
        if self.headIndex >= self.tailIndex:
            print("[Err] Queue Pop")
            return "Err"
        item = self.vec[self.headIndex]
        self.headIndex = self.headIndex + 1
        return item


def main(inputData: InputData) -> OutputData:
    """This function must exist with the specified input and output
    arguments for the submission to work"""

    workshops: List[WorkShop] = []
    for id in range(inputData.N):
        workshops.append(WorkShop(id))

    # Count the earliest and latest time that one workshop can enter
    for wid in range(inputData.M):
        window = inputData.windows[wid]
        workshop = workshops[window.workshopIndex]
        minTi = wid
        maxTi = inputData.L * inputData.M + wid
        if minTi < workshop.minTi:
            workshop.minTi = minTi
        if maxTi > workshop.maxTi:
            workshop.maxTi = maxTi

    # the longest possible window passage
    widOfTi = []
    for loopIndex in range(inputData.L + 1):
        for wid in range(inputData.M):
            widOfTi.append(wid)

    for wid in range(inputData.M, inputData.W):
        window = inputData.windows[wid]
        workshop = workshops[window.workshopIndex]
        widOfTi.append(wid)
        # improvable minTi and maxTi
        minTi = len(widOfTi) - 1
        if minTi < workshop.minTi:
            workshop.minTi = minTi
        if window.canSelfLoop:
            for loopIndex in range(inputData.L):
                widOfTi.append(wid)
        maxTi = len(widOfTi) - 1
        if maxTi > workshop.maxTi:
            workshop.maxTi = maxTi

    if DEBUG:
        for workshop in workshops:
            print(workshop.index)
    workshops.sort(key=lambda de: de.minTi)
    if DEBUG:
        for workshop in workshops:
            print(workshop.index)

    # store the devices on the core production line
    isDeviceInPipeline = [False] * inputData.D
    for eid in inputData.pipeline.edgeIndexs:
        edge = inputData.edges[eid]
        isDeviceInPipeline[edge.sendDevice] = True
        isDeviceInPipeline[edge.recvDevice] = True

    # Make statistics of the workshop area where
    # Collect statistics on the workshop region that support a certain
    # type of equipment in the workshop.

    for rid in range(inputData.R):
        region = inputData.regions[rid]
        nid = region.workshopIndex
        # workshop index in workshops(list) will change due to sort
        workshop = workshops[nid]
        if region.energyType == 0:
            # this can be changed by a list of rid
            workshop.anyRidOfEngine[0].append(rid)
            workshop.anyRidOfEngine[1].append(rid)
        elif region.energyType == 1:
            workshop.anyRidOfEngine[0].append(rid)
        elif region.energyType == 2:
            workshop.anyRidOfEngine[1].append(rid)
        elif region.energyType == 3:
            workshop.anyRidOfEngine[2].append(rid)
        elif region.energyType == 4:
            workshop.anyRidOfEngine[2].append(rid)
    
    # delete the duplicate area in anyRidOfEngine.
    # for workshop in workshops:
    #     for i in range(len(workshop.anyRidOfEngine)):
    #         workshop.anyRidOfEngine[i] = list(set(workshop.anyRidOfEngine[i]))

    # parse flowchart from edge-representation to node-representation
    # e.g nextEdgeMgr[0] stores the 0-th node's outgoing edge 
    nextEdgeMgr = []
    prevEdgeMgr = []
    for did in range(inputData.D):
        nextEdgeMgr.append([])
        prevEdgeMgr.append([])

    for eid in range(inputData.E):
        edge = inputData.edges[eid]
        nextEdgeMgr[edge.sendDevice].append(eid)
        prevEdgeMgr[edge.recvDevice].append(eid)

    queue = Queue()
    # count in edge number for each node
    inCnt = [0] * inputData.D
    # push starting nodes in queue (those don't have incoming edge)
    for did in range(inputData.D):
        inCnt[did] = len(prevEdgeMgr[did])
        if inCnt[did] == 0:
            queue.Push(did)

    # distribute which area for each device
    ridOfDid = [[] for i in range(inputData.D)]
    minTiOfDid = [0] * inputData.D

    # pid is the order of the device on the core production line
    pid = 0
    preTi = 0
    # window scheme for core production line
    widOfPid = [Config.MAX_U32] * (inputData.pipeline.edgeNum + 1)

    while not queue.IsEmpty():
        # current device
        curDid = queue.Pop()
        if isDeviceInPipeline[curDid]:
            startTi = minTiOfDid[curDid]
            engineType = inputData.devices[curDid].engineType
            # ...
            if pid != 0:
                edge = inputData.edges[inputData.pipeline.edgeIndexs[pid - 1]]
                startTi = max(startTi, preTi + (edge.type == 0))

            for ti in range(startTi, len(widOfTi)):
                wid = widOfTi[ti]
                window = inputData.windows[wid]
                # if the window doesn't support pre-processing of the device of this engine type, i.e window selection constraint
                if not window.enginesSupport[engineType]:
                    continue
                workshop = workshops[window.workshopIndex]

                # if the area doesn't support energy of the device of this engine type, i.e device installation constraint
                if len(workshop.anyRidOfEngine[engineType])==0:
                    continue

                # select this window for the window scheme of the core production line
                widOfPid[pid] = wid
                pid = pid + 1
                preTi = ti

                # put current device to those region/areas
                ridOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                break
        else:
            engineType = inputData.devices[curDid].engineType
            for i in range(inputData.N):
                workshop = workshops[i]
                if len(workshop.anyRidOfEngine[engineType])==0:
                    continue

                if workshop.maxTi >= minTiOfDid[curDid]:
                    ridOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                    break
        if len(ridOfDid[curDid]) == 0:
            print("wrong in %d" % curDid)
            exit()
        # record the workshop where the current device installed
        workshop = workshops[inputData.regions[ridOfDid[curDid][0]].workshopIndex]
        for eid in nextEdgeMgr[curDid]:
            edge = inputData.edges[eid]
            curDid = edge.recvDevice
            if edge.type == 0:
                requestTi = workshop.minTi + 1
            else:
                requestTi = workshop.minTi
            minTiOfDid[curDid] = Max(minTiOfDid[curDid], requestTi)
            inCnt[curDid] = inCnt[curDid] - 1
            if inCnt[curDid] == 0:
                queue.Push(curDid)

    possible_regionIndexs = [p for p in itertools.product(*ridOfDid)]
    # n_selected  = min(int(len(possible_regionIndexs)/10), 100)
    # selected_regionIndexs = random.sample(possible_regionIndexs, n_selected)

    Costs = [float("inf")] * len(possible_regionIndexs)
    for idx, region_device in enumerate(possible_regionIndexs):
        outputData = OutputData(
            deviceNum=inputData.D,
            regionIndexs=region_device,
            stepNum=inputData.pipeline.edgeNum + 1,
            timeWindowIndexs=widOfPid,
        )
        Costs[idx] = computeCost(inputData, outputData)

    cost = min(Costs)
    best_regionIndex = Costs.index(cost)
    outputData_FV = OutputData(
            deviceNum=inputData.D,
            regionIndexs=possible_regionIndexs[best_regionIndex],
            stepNum=inputData.pipeline.edgeNum + 1,
            timeWindowIndexs=widOfPid,
        )

    return outputData_FV


if __name__ == "__main__":
    # The following is only used for local tests
    import sys
    import constants

    # inputData = InputData.from_file(sys.argv[1])
    inputData = InputData.from_file('./sample/sample.in')
    outputData = main(inputData)
    outputData.print()
    print(computeCost(inputData, outputData))
    # outputData = constants.sample_output
    # outputData.print()
    # print(computeCost(inputData, outputData))
    