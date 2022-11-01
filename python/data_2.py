#!/usr/bin/env python3
from typing import List
from dataclasses import dataclass
import os.path
import sys

from config import Config, Debug
from reader import IntReader


@dataclass
class Grade:
    id: int
    score: float = 0
    status: str = "failed"
    msg: str = Config.ERR_MSG_UNABLE_JUDGED


@dataclass
class Energy:
    processTime: int

    @classmethod
    def from_reader(cls, reader: IntReader):
        processTime = reader.Read()
        return cls(processTime)


@dataclass
class Region:
    workshopIndex: int
    energyType: int
    energyLimit: int

    @classmethod
    def from_reader(cls, reader: IntReader):
        workshopIndex = reader.Read()
        energyType = reader.Read()
        energyLimit = reader.Read()
        return cls(
            workshopIndex=workshopIndex, energyType=energyType, energyLimit=energyLimit
        )


@dataclass
class Window:
    canSelfLoop: int
    workshopIndex: int
    costFactor: int
    engineAbles: List[int]

    @classmethod
    def from_reader(cls, reader: IntReader):
        canSelfLoop = reader.Read()
        workshopIndex = reader.Read()
        costFactor = reader.Read()
        engineAbles = []
        for i in range(Config.ENGINE_NUM):
            engineAbles.append(reader.Read())
        return cls(
            canSelfLoop=canSelfLoop,
            workshopIndex=workshopIndex,
            costFactor=costFactor,
            engineAbles=engineAbles,
        )


@dataclass
class Device:
    engineType: int
    energyCosts: List[int]
    energyUses: List[int]

    @classmethod
    def from_reader(cls, reader: IntReader):
        engineType = reader.Read()
        energyCosts = []
        for i in range(Config.ENERGY_NUM):
            energyCosts.append(reader.Read())
        energyUses = []
        for i in range(Config.ENERGY_NUM):
            energyUses.append(reader.Read())

        return cls(
            engineType=engineType, energyCosts=energyCosts, energyUses=energyUses
        )


@dataclass
class Edge:
    type: int
    sendDeviceIndex: int
    recvDeviceIndex: int

    @classmethod
    def from_reader(cls, reader: IntReader):
        type = reader.Read()
        sendDeviceIndex = reader.Read()
        recvDeviceIndex = reader.Read()
        return cls(
            type=type, sendDeviceIndex=sendDeviceIndex, recvDeviceIndex=recvDeviceIndex
        )


@dataclass
class Pipeline:
    k: int
    edgeNum: int
    edgeIndexs: List[int]

    @classmethod
    def from_reader(cls, reader: IntReader):
        k = reader.Read()
        edgeNum = reader.Read()
        if edgeNum > Config.MAX_EDGE_NUM:
            return
        edgeIndexs = []
        for i in range(edgeNum):
            edgeIndexs.append(reader.Read())
        return cls(k=k, edgeNum=edgeNum, edgeIndexs=edgeIndexs)


@dataclass
class InputData:
    energys: List[Energy]
    N: int
    R: int
    regions: List[Region]
    L: int
    M: int
    W: int
    windows: List[Window]
    D: int
    devices: List[Device]
    E: int
    edges: List[Edge]
    T: int
    pipelines: List[Pipeline]

    @classmethod
    def from_file(cls, path: str):
        """Read input data from a pathg"""
        reader = IntReader(path)
        data = cls.from_reader(reader)
        return data

    @classmethod
    def from_reader(cls, reader: IntReader) -> "InputData":
        energys = []
        for i in range(Config.ENERGY_NUM):
            energy = Energy.from_reader(reader)
            energys.append(energy)

        N = reader.Read()
        R = reader.Read()
        if R > Config.MAX_REGION_NUM:
            raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
        regions = []
        for i in range(R):
            region = Region.from_reader(reader)
            regions.append(region)

        L = reader.Read() + 1
        M = reader.Read()

        W = reader.Read()
        if W > Config.MAX_WINDOW_NUM:
            raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
        windows = []
        for i in range(W):
            window = Window.from_reader(reader)
            windows.append(window)

        D = reader.Read()
        if D > Config.MAX_DEVICE_NUM:
            raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
        devices = []
        for i in range(D):
            device = Device.from_reader(reader)
            devices.append(device)

        E = reader.Read()
        if E > Config.MAX_EDGE_NUM:
            raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
        edges = []
        for i in range(E):
            edge = Edge.from_reader(reader)
            edges.append(edge)

        T = reader.Read()
        if T > Config.MAX_PIPELINE_NUM:
            raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
        pipelines = []
        for i in range(T):
            pipeline = Pipeline.from_reader(reader)
            pipelines.append(pipeline)

        return cls(
            energys=energys,
            N=N,
            R=R,
            regions=regions,
            L=L,
            M=M,
            W=W,
            windows=windows,
            D=D,
            devices=devices,
            E=E,
            edges=edges,
            T=T,
            pipelines=pipelines,
        )

    def Write(self):
        print(self.energys)
        print(self.N)
        print(self.R)
        print(self.regions)
        print(self.L)
        print(self.M)
        print(self.W)
        print(self.windows)
        print(self.D)
        print(self.devices)
        print(self.E)
        print(self.edges)
        print(self.pipelines)
        return


@dataclass
class OutputData:
    deviceNum: int
    regionIndexs: List[int]
    pipelineNum: int
    widMgr: List[List[int]]

    @classmethod
    def from_file(cls, path: str):
        """Read input data from a pathg"""
        reader = IntReader(path)
        data = OutputData.from_reader(reader)
        return data

    @classmethod
    def from_reader(cls, reader: IntReader) -> "OutputData":
        deviceNum = reader.Read()
        if deviceNum >= Config.MAX_DEVICE_NUM:
            raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
        regionIndexs = []
        for i in range(deviceNum):
            regionIndex = reader.Read()
            regionIndexs.append(regionIndex)

        widMgr = []
        pipelineNum = reader.Read()
        for i in range(pipelineNum):
            stepNum = reader.Read()
            if stepNum >= Config.MAX_STEP_NUM:
                raise ValueError(Config.ERR_MSG_OUTPUT_NUMBER_VIOLATION)
            wids = []
            for i in range(stepNum):
                wid = reader.Read()
                wids.append(wid)
            widMgr.append(wids)

        return cls(
            deviceNum=deviceNum,
            regionIndexs=regionIndexs,
            pipelineNum=pipelineNum,
            widMgr=widMgr,
        )

    def print(self, fh=sys.stdout):
        def PrintVec(vec):
            fh.write(" ".join(str(el) for el in vec) + "\n")

        fh.write(f"{self.deviceNum}\n")
        PrintVec(self.regionIndexs)

        fh.write(f"{self.pipelineNum}\n")
        for row in self.widMgr:
            fh.write(f"{len(row)} ")
            PrintVec(row)


class InputDataMgr:
    @staticmethod
    def from_folder(path: str) -> List[InputData]:
        datas = []
        for caseId in range(Config.CASE_NUM):
            data = InputData.from_file(os.path.join(
                path, "case" + str(caseId) + ".in"))
            datas.append(data)
        return datas


class OutputDataMgr:
    @staticmethod
    def from_folder(path) -> List[OutputData]:
        datas = []
        for caseId in range(Config.CASE_NUM):
            data = OutputData.from_file(
                os.path.join(path, "case" + str(caseId) + ".out")
            )
            datas.append(data)
        return datas
