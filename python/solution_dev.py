import sys
# import itertools
from data import InputData, OutputData, Config, Edge
from typing import List

DEBUG = False

ENGINE_ENERGY = [[0, 1], [2], [3, 4]]


class WorkShop:
    def __init__(self):
        self.regions: List[int] = []
        return

    def add_region(self, rid):
        self.regions.append(rid)
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


def computeWindowProcessTime(inputData, outputData, deviceOnCoreProductionLine):
    # do not using * since list is not elementary type, * will make shallow copy
    windowProcessTimeRaw: List[List[int]] = [[0] for i in range(inputData.W)]
    for id, did in enumerate(deviceOnCoreProductionLine):
        wid = outputData.timeWindowIndexs[id]
        rid = outputData.regionIndexs[did]
        energyType = inputData.regions[rid].energyType
        processTime = inputData.energys[energyType].processTime
        windowProcessTimeRaw[wid].append(processTime)
    windowProcessTime = list(map(max, windowProcessTimeRaw))
    return windowProcessTime


def computeWindowEntryTimes(inputData, outputData, edgeOnCoreProductionLine):
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
    return windowEntryTimes


def computeMaxWindowScheme(inputData: InputData):
    # the longest possible window passage
    widOfTi: List[int] = []
    for loopIndex in range(inputData.L + 1):
        for wid in range(inputData.M):
            widOfTi.append(wid)
    id = wid + 1
    for wid in range(id, inputData.W):
        window = inputData.windows[wid]
        if window.canSelfLoop:
            widOfTi += [wid] * (inputData.L + 1)
        else:
            widOfTi.append(wid)
    return widOfTi


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
    edgeOnCoreProductionLine = parseEdgeOnCoreProductionLine(inputData)
    deviceOnCoreProductionLine = parseDeviceOnCoreProductionLine(
        edgeOnCoreProductionLine)
    # find out window process time (device -> area -> energy -> processTime)
    windowProcessTime = computeWindowProcessTime(
        inputData, outputData, deviceOnCoreProductionLine)
    # compute entry time
    windowEntryTimes = computeWindowEntryTimes(
        inputData, outputData, edgeOnCoreProductionLine)

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
    # region
    if DEBUG:
        print('areaMatchingCost:\t{}'.format(areaMatchingCost))
        print('windowPresetCost:\t{}'.format(windowPresetCost))
        print('windowProcessCost:\t{}'.format(windowProcessCost))
        print('windowProcessTime:\t{}'.format(windowProcessTime))
        print('windowEntryTimes:\t{}'.format(windowEntryTimes))
        print('windowMatchingCost:\t{}'.format(windowMatchingCost))
    # endregion
    return totalCost


def main(inputData: InputData) -> OutputData:
    """This function must exist with the specified input and output
    arguments for the submission to work"""

    widOfTi = computeMaxWindowScheme(inputData)
    print(widOfTi)
    edgeOnCoreProductionLine = parseEdgeOnCoreProductionLine(inputData)
    deviceOnCoreProductionLine = parseDeviceOnCoreProductionLine(
        edgeOnCoreProductionLine)

    tiOfPid = [Config.MAX_U32] * (inputData.pipeline.edgeNum + 1)

    # initialize workshops
    workshops: List[WorkShop] = [WorkShop() for i in range(inputData.N)]
    for rid, region in enumerate(inputData.regions):
        workshops[region.workshopIndex].add_region(rid)

    # store workshop id for each device
    workshopIndexOfDevice = [Config.MAX_U32] * inputData.D
    # store region candidates for each device
    ridsOfDid = [[] for i in range(inputData.D)]

    # find the first-most window scheme
    print(deviceOnCoreProductionLine)
    print(edgeOnCoreProductionLine)
    left_ti = 0
    for pid, did in enumerate(deviceOnCoreProductionLine):
        device = inputData.devices[did]
        for ti in range(left_ti, len(widOfTi)):
            window = inputData.windows[widOfTi[ti]]
            # assign once encounter
            if window.enginesSupport[device.engineType]:
                tiOfPid[pid] = ti
                break
                # for rid in workshop.regions:
                #     region = inputData.regions[rid]
                #     if region.energyType in ENGINE_ENERGY[device.engineType]:
                #         ridsOfDid[did].append(rid)
        # if not last device
        if pid != inputData.pipeline.edgeNum:
            left_ti = ti + (edgeOnCoreProductionLine[pid].type == 0)
    print(tiOfPid)

    # reverse greedy
    right_ti = len(widOfTi)
    for pid, did in enumerate(reversed(deviceOnCoreProductionLine)):
        pid = inputData.pipeline.edgeNum - pid
        left_ti = tiOfPid[pid]
        minCoeff = float('inf')
        best_ti = left_ti
        # find ti from right to left to ensure a larger interval for next device
        for ti in range(right_ti - 1, left_ti - 1, -1):
            window = inputData.windows[widOfTi[ti]]
            if window.costFactor < minCoeff:  # strict less since we prefer the rightmost one to ensure a larger interval
                minCoeff = window.costFactor
                best_ti = ti
            tiOfPid[pid] = best_ti
            if pid != 0:
                # right_ti of red edge doesn't shrink
                right_ti = best_ti + \
                    (edgeOnCoreProductionLine[pid - 1].type == 1)

    print(tiOfPid)

    # print(ridsOfDid)
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
    ridsOfDid = [[] for i in range(inputData.D)]
    minTiOfDid = [0] * inputData.D

    # pid is the order of the device on the core production line
    pid = 0
    preTi = 0
    # window scheme for core production line

    while not queue.IsEmpty():
        # current device
        curDid = queue.Pop()
        # print(curDid)
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
                if len(workshop.anyRidOfEngine[engineType]) == 0:
                    continue

                # select this window for the window scheme of the core production line
                widOfPid[pid] = wid
                pid = pid + 1
                preTi = ti

                # put current device to those region/areas
                ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                break
        else:
            engineType = inputData.devices[curDid].engineType
            for i in range(inputData.N):
                workshop = workshops[i]
                if len(workshop.anyRidOfEngine[engineType]) == 0:
                    continue

                if workshop.maxTi >= minTiOfDid[curDid]:
                    ridsOfDid[curDid] = workshop.anyRidOfEngine[engineType]
                    break
        if len(ridsOfDid[curDid]) == 0:
            print("wrong in %d" % curDid)
            exit()
        # record the workshop where the current device installed
        workshop = workshops[inputData.regions[ridsOfDid[curDid]
                                               [0]].workshopIndex]
        for eid in nextEdgeMgr[curDid]:
            edge = inputData.edges[eid]
            curDid = edge.recvDevice
            if edge.type == 0:
                requestTi = workshop.minTi + 1
            else:
                requestTi = workshop.minTi
            minTiOfDid[curDid] = max(minTiOfDid[curDid], requestTi)
            inCnt[curDid] = inCnt[curDid] - 1
            if inCnt[curDid] == 0:
                queue.Push(curDid)

    # print(ridsOfDid)

    # possible_regionIndexs = [p for p in itertools.product(*ridsOfDid)]

    ridOfDid = []
    for did, rids in enumerate(ridsOfDid):
        if not isDeviceInPipeline[did]:
            InstallCosts = [inputData.regions[rid].energyType for rid in rids]
            idOfMinInstallCost = min(
                range(len(InstallCosts)), key=InstallCosts.__getitem__)
            ridOfDid.append(rids[idOfMinInstallCost])
        else:
            ProcessTime = [
                inputData.energys[inputData.regions[rid].energyType].processTime for rid in rids]
            idOfMinProcessTime = min(
                range(len(ProcessTime)), key=ProcessTime.__getitem__)
            ridOfDid.append(rids[idOfMinProcessTime])

    print(ridOfDid)
    outputData_FV = OutputData(
        deviceNum=inputData.D,
        regionIndexs=ridOfDid,
        stepNum=inputData.pipeline.edgeNum + 1,
        timeWindowIndexs=widOfPid,
    )

    return outputData_FV

    # print(possible_regionIndexs)
    # n_selected  = min(int(len(possible_regionIndexs)/10), 100)
    # selected_regionIndexs = random.sample(possible_regionIndexs, n_selected)

    # Costs = [float("inf")] * len(possible_regionIndexs)
    # for idx, region_device in enumerate(possible_regionIndexs):
    #     outputData = OutputData(
    #         deviceNum=inputData.D,
    #         regionIndexs=region_device,
    #         stepNum=inputData.pipeline.edgeNum + 1,
    #         timeWindowIndexs=widOfPid,
    #     )
    #     Costs[idx] = computeCost(inputData, outputData)

    # cost = min(Costs)
    # best_regionIndex = Costs.index(cost)
    # outputData_FV = OutputData(
    #         deviceNum=inputData.D,
    #         regionIndexs=possible_regionIndexs[best_regionIndex],
    #         stepNum=inputData.pipeline.edgeNum + 1,
    #         timeWindowIndexs=widOfPid,
    #     )

    # return outputData_FV


if __name__ == "__main__":
    # The following is only used for local tests
    import sys
    # import constants

    # inputData = InputData.from_file(sys.argv[1])
    inputData = InputData.from_file('./sample/sample.in')
    outputData = main(inputData)
    outputData.print()
    print(computeCost(inputData, outputData))
    # outputData = constants.sample_output
    # print('sample_output')
    # outputData.print()
    # print(computeCost(inputData, outputData))

    # for i in range(1,11,1):
    #     outputData = eval('constants.sample_output{}'.format(i))
    #     print('sample_output{}'.format(i))
    #     # outputData.print()
    #     print(computeCost(inputData, outputData))
