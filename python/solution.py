import sys
from data import InputData, OutputData, Config
from typing import List


def Max(a, b):
    if a > b:
        return a
    else:
        return b


class WorkShop:
    def __init__(self, index):
        self.index = index
        self.minTi = Config.MAX_U32
        self.maxTi = 0
        self.anyRidOfEngine = [Config.MAX_U32] * Config.ENGINE_TYPE_NUM
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

    for workshop in workshops:
        print(workshop.index)
    workshops.sort(key=lambda de: de.minTi)
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
            workshop.anyRidOfEngine[0] = rid
            workshop.anyRidOfEngine[1] = rid
        elif region.energyType == 1:
            workshop.anyRidOfEngine[0] = rid
        elif region.energyType == 2:
            workshop.anyRidOfEngine[1] = rid
        elif region.energyType == 3:
            workshop.anyRidOfEngine[2] = rid
        elif region.energyType == 4:
            workshop.anyRidOfEngine[2] = rid

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
    ridOfDid = [Config.MAX_U32] * inputData.D
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
                rid = workshop.anyRidOfEngine[engineType]
                # if the area doesn't support energy of the device of this engine type, i.e device installation constraint
                if rid == Config.MAX_U32:
                    continue

                # put current device to this region/area
                ridOfDid[curDid] = rid
                # select this window for the window scheme of the core production line
                widOfPid[pid] = wid
                pid = pid + 1
                preTi = ti
                break
        else:
            engineType = inputData.devices[curDid].engineType
            for i in range(inputData.N):
                workshop = workshops[i]
                rid = workshop.anyRidOfEngine[engineType]
                if rid == Config.MAX_U32:
                    continue
                if workshop.maxTi >= minTiOfDid[curDid]:
                    ridOfDid[curDid] = rid
                    break
        if ridOfDid[curDid] == Config.MAX_U32:
            print("wrong in %d" % curDid)
            exit()
        # record the workshop where the current device installed
        workshop = workshops[inputData.regions[ridOfDid[curDid]].workshopIndex]
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

    outputData = OutputData(
        deviceNum=inputData.D,
        regionIndexs=ridOfDid,
        stepNum=inputData.pipeline.edgeNum + 1,
        timeWindowIndexs=widOfPid,
    )
    return outputData


if __name__ == "__main__":
    # The following is only used for local tests
    import sys

    # inputData = InputData.from_file(sys.argv[1])
    inputData = InputData.from_file('./sample/sample.in')
    outputData = main(inputData)
    outputData.print()
