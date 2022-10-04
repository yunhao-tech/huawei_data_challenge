import sys
import numpy as np
# from copy import deepcopy
# from itertools import islice, product
from data import InputData, OutputData, Config, Edge
from typing import List

DEBUG = False
alpha = 0.5
beta = 0.5


class WorkShop:
    def __init__(self, index):
        self.index = index
        self.minTi = Config.MAX_U32
        self.maxTi = 0
        self.anyRidOfEngine = [[] for i in range(Config.ENGINE_TYPE_NUM)]
        '''第i个元素储存支持第i种设备的area id(rid)，可以是列表（即多个满足条件的area）'''
        self.energyTypes = []
        return

    def __str__(self) -> str:
        return self.energyTypes.__str__()


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


def parseEdgeOnCoreProductionLine(inputData: InputData):
    edgeOnCoreProductionLine: List[Edge] = []

    for eid in inputData.pipeline.edgeIndexs:
        edge = inputData.edges[eid]
        edgeOnCoreProductionLine.append(edge)
    return edgeOnCoreProductionLine


def parseDeviceOnCoreProductionLine(edgeOnCoreProductionLine: List[Edge]):
    deviceOnCoreProductionLine: List[int] = []
    for edge in edgeOnCoreProductionLine:
        deviceOnCoreProductionLine.append(edge.sendDevice)
    deviceOnCoreProductionLine.append(edge.recvDevice)
    return deviceOnCoreProductionLine


def computeWindowEntryTimes(inputData: InputData, timeWindowIndexs: List[int]):
    edgeOnCoreProductionLine: List[Edge] = []

    for eid in inputData.pipeline.edgeIndexs:
        edge = inputData.edges[eid]
        edgeOnCoreProductionLine.append(edge)
        # print("edge {} type {}".format(eid, edge.type))

    deviceOnCoreProductionLine: List[int] = []
    for edge in edgeOnCoreProductionLine:
        deviceOnCoreProductionLine.append(edge.sendDevice)
    deviceOnCoreProductionLine.append(edge.recvDevice)

    windowEntryTimes: List[int] = [0] * inputData.W
    for id, edge in enumerate(edgeOnCoreProductionLine):
        wid = timeWindowIndexs[id]
        wid_next = timeWindowIndexs[id + 1]
        windowEntryTimes[wid] += 1
        # if two consecutive windows are the same and the edge is red, i.e collabarative relation
        # only count once entry time
        if wid == wid_next and edge.type == 1:
            windowEntryTimes[wid] -= 1  # minus one since count one more time
            # print("collaboration in window {}".format(wid))
    # Don't forget the last device
    windowEntryTimes[wid_next] += 1
    return windowEntryTimes, deviceOnCoreProductionLine


def computeCost(inputData: InputData, outputData: OutputData):
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
    # do not using * since list is not elementary type, * will make shallow copy
    windowProcessTimeRaw: List[List[int]] = [[0] for i in range(inputData.W)]
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
            windowEntryTimes[wid] -= 1  # minus one since count one more time
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


def main(inputData: InputData) -> OutputData:
    """This function must exist with the specified input and output
    arguments for the submission to work"""

    # workshops initialization
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
    # region
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
    # endregion

    # store the devices on the core production line
    isDeviceInPipeline = [False] * inputData.D
    for eid in inputData.pipeline.edgeIndexs:
        edge = inputData.edges[eid]
        isDeviceInPipeline[edge.sendDevice] = True
        isDeviceInPipeline[edge.recvDevice] = True

    # Collect statistics on the workshop region that support a certain
    # type of equipment in the workshop.
    # region
    for rid in range(inputData.R):
        region = inputData.regions[rid]
        nid = region.workshopIndex
        # workshop index in workshops(list) will change due to sort
        workshop = workshops[nid]
        workshop.energyTypes.append(region.energyType)
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

    # for workshop in workshops:
    #     print(workshop)
    # endregion

    # parse flowchart from edge-representation to node-representation
    # region
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
    # endregion

    # sort workshops
    # region
    # for workshop in workshops:
    #     print(workshop.index, workshop.minTi, workshop.maxTi)

    workshopIndices_sorted_by_minTi = np.argsort(
        [workshop.minTi for workshop in workshops])
    workshopIndices_sorted_by_maxTi = np.argsort(
        [workshop.maxTi for workshop in workshops])

    if DEBUG:
        print('workshops')
        for workshop in workshops:
            print(workshop.index, end=' ')
        print('')
    if DEBUG:
        print('workshops_sorted_by_minTi')
        for id in workshopIndices_sorted_by_minTi:
            print(workshops[id].index, end=' ')
        print('')
    if DEBUG:
        print('workshops_sorted_by_maxTi')
        for id in workshopIndices_sorted_by_maxTi:
            print(workshops[id].index, end=' ')
        print('')
    # endregion

    # topological sort
    # region
    queue = Queue()
    # count in edge number for each node
    inCnt = [0] * inputData.D
    # push starting nodes in queue (those don't have incoming edge)
    for did in range(inputData.D):
        inCnt[did] = len(prevEdgeMgr[did])
        if inCnt[did] == 0:
            queue.Push(did)
    while not queue.IsEmpty():
        # current device
        curDid = queue.Pop()
        for eid in nextEdgeMgr[curDid]:
            edge = inputData.edges[eid]
            curDid = edge.recvDevice
            inCnt[curDid] = inCnt[curDid] - 1
            if inCnt[curDid] == 0:
                queue.Push(curDid)

    # print(queue.vec)
    # endregion

    # distribute which area for each device
    ridsOfDid = [[] for i in range(inputData.D)]
    # window scheme for core production line
    widOfPid = [Config.MAX_U32] * (inputData.pipeline.edgeNum + 1)
    # timestamp for each device
    tiOfDid = [None] * inputData.D

    # time-first forward greedy
    # pid is the order of the device on the core production line
    pid = 0
    # previous pipeline device's Ti
    preTi = 0
    minTiOfDid = [0] * inputData.D

    for curDid in queue.vec:
        if isDeviceInPipeline[curDid]:
            startTi = minTiOfDid[curDid]
            engineType = inputData.devices[curDid].engineType

            if pid != 0:  # except the first one
                edge = inputData.edges[inputData.pipeline.edgeIndexs[pid - 1]]
                # startTi computes the earliest possible timestamp for device on pipeline
                startTi = max(startTi, preTi + (edge.type == 0))

            for ti in range(startTi, len(widOfTi)):
                wid = widOfTi[ti]
                window = inputData.windows[wid]
                # if the window doesn't support pre-processing of the device of this engine type, i.e window selection constraint
                if not window.enginesSupport[engineType]:
                    continue

                workshop = workshops[window.workshopIndex]
                # print(window.workshopIndex, workshop.index)
                # if the area doesn't support energy of the device of this engine type, i.e device installation constraint
                if len(workshop.anyRidOfEngine[engineType]) == 0:
                    continue

                # select this window for the window scheme of the core production line
                widOfPid[pid] = wid
                pid = pid + 1
                tiOfDid[curDid] = ti
                preTi = ti

                # put current device to those region/areas
                ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                break
        else:
            engineType = inputData.devices[curDid].engineType
            for i in workshopIndices_sorted_by_minTi:
                workshop = workshops[i]
                if len(workshop.anyRidOfEngine[engineType]) == 0:
                    continue

                if workshop.maxTi >= minTiOfDid[curDid]:
                    ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                    tiOfDid[curDid] = workshop.minTi
                    break
        if len(ridsOfDid[curDid]) == 0:
            print("wrong in %d" % curDid)
            exit()
        # record the workshop where the current device installed(can be skipped)
        workshop = workshops[inputData.regions[ridsOfDid[curDid]
                                               [0]].workshopIndex]
        # update minTi for the next device
        for eid in nextEdgeMgr[curDid]:
            edge = inputData.edges[eid]
            postDid = edge.recvDevice
            if edge.type == 0:
                requestTi = workshop.minTi + 1
            else:
                requestTi = workshop.minTi
            # minTiOfDid stores the earliest possible timestamp given by the topological order
            # the requestTi is computed from workshop.minTi, as if every time the device enter the workshop is the first time (no need to count the max loopback constraint)
            minTiOfDid[postDid] = max(minTiOfDid[postDid], requestTi)

    # print(queue.vec)
    # print(tiOfDid)
    # print(pid)

    for epoch in range(1):
        # cost-first backward greedy
        pid = inputData.pipeline.edgeNum  # start from right most
        # post pipeline device's Ti
        postTi = len(widOfTi) - 1  # initialize with the latest timestamp
        maxTiOfDid = [len(widOfTi) - 1] * inputData.D

        for curDid in reversed(queue.vec):
            if isDeviceInPipeline[curDid]:
                endTi = maxTiOfDid[curDid]
                engineType = inputData.devices[curDid].engineType

                if pid != inputData.pipeline.edgeNum:  # except the last one
                    edge = inputData.edges[inputData.pipeline.edgeIndexs[pid - 1]]
                    # endTi computes the latest possible timestamp for device on pipeline
                    endTi = min(endTi, postTi - (edge.type == 0))

                window_old = inputData.windows[widOfPid[pid]]
                # window_candidate = window_old
                for ti in range(endTi, tiOfDid[curDid], -1):
                    wid = widOfTi[ti]
                    window = inputData.windows[wid]
                    # if the window doesn't support pre-processing of the device of this engine type, i.e window selection constraint
                    if not window.enginesSupport[engineType]:
                        continue
                    # cost greedy
                    if window.costFactor > window_old.costFactor:
                        continue

                    workshop = workshops[window.workshopIndex]
                    # if the area doesn't support energy of the device of this engine type, i.e device installation constraint
                    if len(workshop.anyRidOfEngine[engineType]) == 0:
                        continue

                    # select this window for the window scheme of the core production line
                    widOfPid[pid] = wid
                    pid = pid - 1
                    tiOfDid[curDid] = ti
                    postTi = ti

                    # put current device to those region/areas
                    ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                    break
            else:
                workshop_old = workshops[inputData.regions[ridsOfDid[curDid][0]].workshopIndex]
                device = inputData.devices[curDid]
                minInstallCost_old = np.ma.masked_equal(np.array(
                    [device.energyCosts[energyType] for energyType in workshop_old.energyTypes]), 0, copy=False).min()
                # print(minInstallCost_old)
                engineType = inputData.devices[curDid].engineType
                for i in reversed(workshopIndices_sorted_by_maxTi):
                    workshop = workshops[i]
                    if len(workshop.anyRidOfEngine[engineType]) == 0:
                        continue
                    minInstallCost = np.ma.masked_equal(np.array(
                        [device.energyCosts[energyType] for energyType in workshop.energyTypes]), 0, copy=False).min()
                    if minInstallCost > minInstallCost_old:
                        continue
                    if workshop.minTi <= maxTiOfDid[curDid]:
                        ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                        tiOfDid[curDid] = workshop.maxTi
                        break
            if len(ridsOfDid[curDid]) == 0:
                print("wrong in %d" % curDid)
                exit()
            # record the workshop where the current device installed(can be skipped)
            workshop = workshops[inputData.regions[ridsOfDid[curDid]
                                                   [0]].workshopIndex]

            # update maxTi for the next device
            for eid in prevEdgeMgr[curDid]:
                edge = inputData.edges[eid]
                preDid = edge.sendDevice
                if edge.type == 0:
                    requestTi = workshop.maxTi - 1
                else:
                    requestTi = workshop.maxTi
                # maxTiOfDid stores the latest possible timestamp given by the topologival order
                # the requestTi is computed from workshop.maxTi, as if every time the device enter the workshop is the first time (no need to count the maximum loopback)
                maxTiOfDid[preDid] = min(maxTiOfDid[preDid], requestTi)
        # print(tiOfDid)

        # cost-first forward greedy
            # pid is the order of the device on the core production line
        pid = 0
        # previous pipeline device's Ti
        preTi = 0

        for curDid in queue.vec:
            if isDeviceInPipeline[curDid]:
                startTi = minTiOfDid[curDid]
                engineType = inputData.devices[curDid].engineType

                if pid != 0:  # except the first one
                    edge = inputData.edges[inputData.pipeline.edgeIndexs[pid - 1]]
                    # startTi computes the earliest possible timestamp for device on pipeline
                    startTi = max(startTi, preTi + (edge.type == 0))

                for ti in range(startTi, tiOfDid[curDid]):
                    wid = widOfTi[ti]
                    window = inputData.windows[wid]
                    # if the window doesn't support pre-processing of the device of this engine type, i.e window selection constraint
                    if not window.enginesSupport[engineType]:
                        continue
                    # cost greedy
                    if window.costFactor > window_old.costFactor:
                        continue

                    workshop = workshops[window.workshopIndex]
                    # print(window.workshopIndex, workshop.index)
                    # if the area doesn't support energy of the device of this engine type, i.e device installation constraint
                    if len(workshop.anyRidOfEngine[engineType]) == 0:
                        continue

                    # select this window for the window scheme of the core production line
                    widOfPid[pid] = wid
                    pid = pid + 1
                    tiOfDid[curDid] = ti
                    preTi = ti

                    # put current device to those region/areas
                    ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                    break
            else:
                workshop_old = workshops[inputData.regions[ridsOfDid[curDid][0]].workshopIndex]
                device = inputData.devices[curDid]
                minInstallCost_old = np.ma.masked_equal(np.array(
                    [device.energyCosts[energyType] for energyType in workshop_old.energyTypes]), 0, copy=False).min()
                # print(minInstallCost_old)
                engineType = inputData.devices[curDid].engineType
                for i in workshopIndices_sorted_by_minTi:
                    workshop = workshops[i]
                    if len(workshop.anyRidOfEngine[engineType]) == 0:
                        continue
                    minInstallCost = np.ma.masked_equal(np.array(
                        [device.energyCosts[energyType] for energyType in workshop.energyTypes]), 0, copy=False).min()
                    if minInstallCost > minInstallCost_old:
                        continue
                    if workshop.maxTi >= minTiOfDid[curDid]:
                        ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                        tiOfDid[curDid] = workshop.minTi
                        break
            if len(ridsOfDid[curDid]) == 0:
                print("wrong in %d" % curDid)
                exit()
            # record the workshop where the current device installed(can be skipped)
            workshop = workshops[inputData.regions[ridsOfDid[curDid]
                                                   [0]].workshopIndex]
            # update minTi for the next device
            for eid in nextEdgeMgr[curDid]:
                edge = inputData.edges[eid]
                postDid = edge.recvDevice
                if edge.type == 0:
                    requestTi = workshop.minTi + 1
                else:
                    requestTi = workshop.minTi
                # minTiOfDid stores the earliest possible timestamp given by the topological order
                # the requestTi is computed from workshop.minTi, as if every time the device enter the workshop is the first time (no need to count the max loopback constraint)
                minTiOfDid[postDid] = max(minTiOfDid[postDid], requestTi)

    ridOfDid = []
    windowEntryTimes, deviceOnCoreProductionLine = computeWindowEntryTimes(
        inputData, widOfPid)
    for did, rids in enumerate(ridsOfDid):
        InstallCosts = [
            inputData.devices[did].energyCosts[inputData.regions[rid].energyType] for rid in rids]
        if not isDeviceInPipeline[did]:
            idOfMinInstallCost = min(
                range(len(InstallCosts)), key=InstallCosts.__getitem__)
            ridOfDid.append(rids[idOfMinInstallCost])
        else:
            # processTime of a single device in an area/region
            ProcessTime = [
                inputData.energys[inputData.regions[rid].energyType].processTime for rid in rids]
            # processTime of a workshop
            workshop = workshops[inputData.regions[rids[0]].workshopIndex]
            # print(workshop.index)
            processTimes = [
                inputData.energys[energy_id].processTime for energy_id in workshop.energyTypes]
            maxProcessTime = max(processTimes)
            # print(ProcessTime)
            # print(processTimes)
            # print(maxProcessTime)
            wid = widOfPid[deviceOnCoreProductionLine.index(did)]
            WindowEntryTimes = windowEntryTimes[wid]
            MixCost = ((1-alpha) * np.array(ProcessTime) + (alpha * maxProcessTime)) * inputData.K * WindowEntryTimes + ((1 - beta) * np.array(ProcessTime) + beta * maxProcessTime) * \
                inputData.windows[wid].costFactor + np.array(InstallCosts)
            idMinMixCost = np.argmin(MixCost)
            ridOfDid.append(rids[idMinMixCost])

    # print(ridOfDid)
    outputData_FV = OutputData(
        deviceNum=inputData.D,
        regionIndexs=ridOfDid,
        stepNum=inputData.pipeline.edgeNum + 1,
        timeWindowIndexs=widOfPid,
    )
    # raise ValueError(widOfPid)
    # raise ValueError(np.array(tiOfDid)[deviceOnCoreProductionLine])
    return outputData_FV


if __name__ == "__main__":
    # The following is only used for local tests
    # inputData = InputData.from_file(sys.argv[1])
    # inputData = InputData.from_file('./sample/sample.in')
    # inputData = InputData.from_file('./sample/sample_test.in')
    inputData = InputData.from_file('./sample/sample scratch.in')
    outputData = main(inputData)
    outputData.print()
    print(computeCost(inputData, outputData))
